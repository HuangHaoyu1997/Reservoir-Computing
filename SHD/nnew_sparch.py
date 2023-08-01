import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from distutils.util import strtobool

import logging

import h5py
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import errno
import logging
import os
import time
from datetime import timedelta

from torch.optim.lr_scheduler import ReduceLROnPlateau

logger = logging.getLogger(__name__)

class SpikeFunctionBoxcar(torch.autograd.Function):
    """
    Compute surrogate gradient of the spike step function using
    box-car function similar to DECOLLE, Kaiser et al. (2020).
    """

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.gt(0).float()

    def backward(ctx, grad_spikes):
        (x,) = ctx.saved_tensors
        grad_x = grad_spikes.clone()
        grad_x[x <= -0.5] = 0
        grad_x[x > 0.5] = 0
        return grad_x


class SNN(nn.Module):
    """
    A multi-layered Spiking Neural Network (SNN).

    It accepts input tensors formatted as (batch, time, feat). In the case of
    4d inputs like (batch, time, feat, channel) the input is flattened as
    (batch, time, feat*channel).

    The function returns the outputs of the last spiking or readout layer
    with shape (batch, time, feats) or (batch, feats) respectively, as well
    as the firing rates of all hidden neurons with shape (num_layers*feats).
    """

    def __init__(
        self,
        input_shape,
        layer_sizes,
        neuron_type="LIF",
        threshold=1.0,
        dropout=0.0,
        normalization="batchnorm",
        use_bias=False,
        bidirectional=False,
        use_readout_layer=True,
    ):
        super().__init__()

        # Fixed parameters
        self.reshape = True if len(input_shape) > 3 else False
        self.input_size = float(torch.prod(torch.tensor(input_shape[2:])))
        self.batch_size = input_shape[0]
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        self.num_outputs = layer_sizes[-1]
        self.neuron_type = neuron_type
        self.threshold = threshold
        self.dropout = dropout
        self.normalization = normalization
        self.use_bias = use_bias
        self.bidirectional = bidirectional
        self.use_readout_layer = use_readout_layer
        self.is_snn = True

        if neuron_type not in ["LIF", "adLIF", "RLIF", "RadLIF"]:
            raise ValueError(f"Invalid neuron type {neuron_type}")

        # Init trainable parameters
        self.snn = self._init_layers()

    def _init_layers(self):
        snn = nn.ModuleList([])
        input_size = self.input_size
        snn_class = self.neuron_type + "Layer"

        if self.use_readout_layer:
            num_hidden_layers = self.num_layers - 1
        else:
            num_hidden_layers = self.num_layers

        # Hidden layers
        for i in range(num_hidden_layers):
            snn.append(
                globals()[snn_class](
                    input_size=input_size,
                    hidden_size=self.layer_sizes[i],
                    batch_size=self.batch_size,
                    threshold=self.threshold,
                    dropout=self.dropout,
                    normalization=self.normalization,
                    use_bias=self.use_bias,
                    bidirectional=self.bidirectional,
                )
            )
            input_size = self.layer_sizes[i] * (1 + self.bidirectional)

        # Readout layer
        if self.use_readout_layer:
            snn.append(
                ReadoutLayer(
                    input_size=input_size,
                    hidden_size=self.layer_sizes[-1],
                    batch_size=self.batch_size,
                    dropout=self.dropout,
                    normalization=self.normalization,
                    use_bias=self.use_bias,
                )
            )

        return snn

    def forward(self, x):

        # Reshape input tensors to (batch, time, feats) for 4d inputs
        if self.reshape:
            if x.ndim == 4:
                x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])
            else:
                raise NotImplementedError

        # Process all layers
        all_spikes = []
        for i, snn_lay in enumerate(self.snn):
            x = snn_lay(x)
            if not (self.use_readout_layer and i == self.num_layers - 1):
                all_spikes.append(x)

        # Compute mean firing rate of each spiking neuron
        firing_rates = torch.cat(all_spikes, dim=2).mean(dim=(0, 1))

        return x, firing_rates
    
class RadLIFLayer(nn.Module):
    """
    A single layer of adaptive Leaky Integrate-and-Fire neurons with layer-wise
    recurrent connections (RadLIF).
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        batch_size,
        threshold=1.0,
        dropout=0.0,
        normalization="batchnorm",
        use_bias=False,
        bidirectional=False,
    ):
        super().__init__()

        # Fixed parameters
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.batch_size = batch_size
        self.threshold = threshold
        self.dropout = dropout
        self.normalization = normalization
        self.use_bias = use_bias
        self.bidirectional = bidirectional
        self.batch_size = self.batch_size * (1 + self.bidirectional)
        self.alpha_lim = [np.exp(-1 / 5), np.exp(-1 / 25)]
        self.beta_lim = [np.exp(-1 / 30), np.exp(-1 / 120)]
        self.a_lim = [-1.0, 1.0]
        self.b_lim = [0.0, 2.0]
        self.spike_fct = SpikeFunctionBoxcar.apply

        # Trainable parameters
        self.W = nn.Linear(self.input_size, self.hidden_size, bias=use_bias)
        self.V = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.alpha = nn.Parameter(torch.Tensor(self.hidden_size))
        self.beta = nn.Parameter(torch.Tensor(self.hidden_size))
        self.a = nn.Parameter(torch.Tensor(self.hidden_size))
        self.b = nn.Parameter(torch.Tensor(self.hidden_size))

        nn.init.uniform_(self.alpha, self.alpha_lim[0], self.alpha_lim[1])
        nn.init.uniform_(self.beta, self.beta_lim[0], self.beta_lim[1])
        nn.init.uniform_(self.a, self.a_lim[0], self.a_lim[1])
        nn.init.uniform_(self.b, self.b_lim[0], self.b_lim[1])
        nn.init.orthogonal_(self.V.weight)

        # Initialize normalization
        self.normalize = False
        if normalization == "batchnorm":
            self.norm = nn.BatchNorm1d(self.hidden_size, momentum=0.05)
            self.normalize = True
        elif normalization == "layernorm":
            self.norm = nn.LayerNorm(self.hidden_size)
            self.normalize = True

        # Initialize dropout
        self.drop = nn.Dropout(p=dropout)

    def forward(self, x):

        # Concatenate flipped sequence on batch dim
        if self.bidirectional:
            x_flip = x.flip(1)
            x = torch.cat([x, x_flip], dim=0)

        # Change batch size if needed
        if self.batch_size != x.shape[0]:
            self.batch_size = x.shape[0]

        # Feed-forward affine transformations (all steps in parallel)
        Wx = self.W(x)

        # Apply normalization
        if self.normalize:
            _Wx = self.norm(Wx.reshape(Wx.shape[0] * Wx.shape[1], Wx.shape[2]))
            Wx = _Wx.reshape(Wx.shape[0], Wx.shape[1], Wx.shape[2])

        # Compute spikes via neuron dynamics
        s = self._radlif_cell(Wx)

        # Concatenate forward and backward sequences on feat dim
        if self.bidirectional:
            s_f, s_b = s.chunk(2, dim=0)
            s_b = s_b.flip(1)
            s = torch.cat([s_f, s_b], dim=2)

        # Apply dropout
        s = self.drop(s)

        return s

    def _radlif_cell(self, Wx):

        # Initializations
        device = Wx.device
        ut = torch.rand(Wx.shape[0], Wx.shape[2]).to(device)
        wt = torch.rand(Wx.shape[0], Wx.shape[2]).to(device)
        st = torch.rand(Wx.shape[0], Wx.shape[2]).to(device)
        s = []

        # Bound values of the neuron parameters to plausible ranges
        alpha = torch.clamp(self.alpha, min=self.alpha_lim[0], max=self.alpha_lim[1])
        beta = torch.clamp(self.beta, min=self.beta_lim[0], max=self.beta_lim[1])
        a = torch.clamp(self.a, min=self.a_lim[0], max=self.a_lim[1])
        b = torch.clamp(self.b, min=self.b_lim[0], max=self.b_lim[1])

        # Set diagonal elements of recurrent matrix to zero
        V = self.V.weight.clone().fill_diagonal_(0)

        # Loop over time axis
        for t in range(Wx.shape[1]):

            # Compute potential (RadLIF)
            wt = beta * wt + a * ut + b * st
            ut = alpha * (ut - st) + (1 - alpha) * (
                Wx[:, t, :] + torch.matmul(st, V) - wt
            )

            # Compute spikes with surrogate gradient
            st = self.spike_fct(ut - self.threshold)
            s.append(st)

        return torch.stack(s, dim=1)

class ReadoutLayer(nn.Module):
    """
    This function implements a single layer of non-spiking Leaky Integrate and
    Fire (LIF) neurons, where the output consists of a cumulative sum of the
    membrane potential using a softmax function, instead of spikes.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        batch_size,
        dropout=0.0,
        normalization="batchnorm",
        use_bias=False,
    ):
        super().__init__()

        # Fixed parameters
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.batch_size = batch_size
        self.dropout = dropout
        self.normalization = normalization
        self.use_bias = use_bias
        self.alpha_lim = [np.exp(-1 / 5), np.exp(-1 / 25)]

        # Trainable parameters
        self.W = nn.Linear(self.input_size, self.hidden_size, bias=use_bias)
        self.alpha = nn.Parameter(torch.Tensor(self.hidden_size))
        nn.init.uniform_(self.alpha, self.alpha_lim[0], self.alpha_lim[1])

        # Initialize normalization
        self.normalize = False
        if normalization == "batchnorm":
            self.norm = nn.BatchNorm1d(self.hidden_size, momentum=0.05)
            self.normalize = True
        elif normalization == "layernorm":
            self.norm = nn.LayerNorm(self.hidden_size)
            self.normalize = True

        # Initialize dropout
        self.drop = nn.Dropout(p=dropout)

    def forward(self, x):
        # Feed-forward affine transformations (all steps in parallel)
        Wx = self.W(x)

        # Apply normalization
        if self.normalize:
            _Wx = self.norm(Wx.reshape(Wx.shape[0] * Wx.shape[1], Wx.shape[2]))
            Wx = _Wx.reshape(Wx.shape[0], Wx.shape[1], Wx.shape[2])

        # Compute membrane potential via non-spiking neuron dynamics
        out = self._readout_cell(Wx)

        return out

    def _readout_cell(self, Wx):

        # Initializations
        device = Wx.device
        ut = torch.rand(Wx.shape[0], Wx.shape[2]).to(device)
        out = torch.zeros(Wx.shape[0], Wx.shape[2]).to(device)

        # Bound values of the neuron parameters to plausible ranges
        alpha = torch.clamp(self.alpha, min=self.alpha_lim[0], max=self.alpha_lim[1])

        # Loop over time axis
        for t in range(Wx.shape[1]):

            # Compute potential (LIF)
            ut = alpha * ut + (1 - alpha) * Wx[:, t, :]
            out = out + F.softmax(ut, dim=1)

        return out

class SpikingDataset(Dataset):
    """
    Dataset class for the Spiking Heidelberg Digits (SHD) or Spiking Speech Commands (SSC) dataset.
    """

    def __init__(
        self,
        dataset_name,
        data_folder,
        split,
        nb_steps=100,
    ):

        # Fixed parameters
        self.device = "cpu"  # to allow pin memory
        self.nb_steps = nb_steps
        self.nb_units = 700
        self.max_time = 1.4
        self.time_bins = np.linspace(0, self.max_time, num=self.nb_steps)

        # Read data from h5py file
        filename = f"{data_folder}/{dataset_name}_{split}.h5"
        self.h5py_file = h5py.File(filename, "r")
        self.firing_times = self.h5py_file["spikes"]["times"]
        self.units_fired = self.h5py_file["spikes"]["units"]
        self.labels = np.array(self.h5py_file["labels"], dtype=np.int)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):

        times = np.digitize(self.firing_times[index], self.time_bins)
        units = self.units_fired[index]

        x_idx = torch.LongTensor(np.array([times, units])).to(self.device)
        x_val = torch.FloatTensor(np.ones(len(times))).to(self.device)
        x_size = torch.Size([self.nb_steps, self.nb_units])

        x = torch.sparse.FloatTensor(x_idx, x_val, x_size).to(self.device)
        y = self.labels[index]

        return x.to_dense(), y

    def generateBatch(self, batch):

        xs, ys = zip(*batch)
        xs = torch.nn.utils.rnn.pad_sequence(xs, batch_first=True)
        xlens = torch.tensor([x.shape[0] for x in xs])
        ys = torch.LongTensor(ys).to(self.device)

        return xs, xlens, ys


def load_shd_or_ssc(
    dataset_name,
    data_folder,
    split,
    batch_size,
    nb_steps=100,
    shuffle=True,
    workers=0,
):
    """
    This function creates a dataloader for a given split of the SHD or SSC datasets.
    """
    if dataset_name not in ["shd", "ssc"]:
        raise ValueError(f"Invalid dataset name {dataset_name}")

    if split not in ["train", "valid", "test"]:
        raise ValueError(f"Invalid split name {split}")

    if dataset_name == "shd" and split == "valid":
        logging.info("SHD does not have a validation split. Using test split.")
        split = "test"

    dataset = SpikingDataset(dataset_name, data_folder, split, nb_steps)
    logging.info(f"Number of examples in {split} set: {len(dataset)}")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=dataset.generateBatch,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
    )
    return loader

class Experiment:
    """
    Class for training and testing models (ANNs and SNNs) on all four
    datasets for speech command recognition (shd, ssc, hd and sc).
    """

    def __init__(self, args):

        # New model config
        self.model_type = args.model_type
        self.nb_layers = args.nb_layers
        self.nb_hiddens = args.nb_hiddens
        self.pdrop = args.pdrop
        self.normalization = args.normalization
        self.use_bias = args.use_bias
        self.bidirectional = args.bidirectional

        # Training config
        self.use_pretrained_model = args.use_pretrained_model
        self.only_do_testing = args.only_do_testing
        self.load_exp_folder = args.load_exp_folder
        self.new_exp_folder = args.new_exp_folder
        self.dataset_name = args.dataset_name
        self.data_folder = args.data_folder
        self.log_tofile = args.log_tofile
        self.save_best = args.save_best
        self.batch_size = args.batch_size
        self.nb_epochs = args.nb_epochs
        self.start_epoch = args.start_epoch
        self.lr = args.lr
        self.scheduler_patience = args.scheduler_patience
        self.scheduler_factor = args.scheduler_factor
        self.use_regularizers = args.use_regularizers
        self.reg_factor = args.reg_factor
        self.reg_fmin = args.reg_fmin
        self.reg_fmax = args.reg_fmax
        self.use_augm = args.use_augm

        # Initialize logging and output folders
        self.init_exp_folders()
        self.init_logging()

        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"\nDevice is set to {self.device}\n")

        # Initialize dataloaders and model
        self.init_dataset()
        self.init_model()

        # Define optimizer
        self.opt = torch.optim.Adam(self.net.parameters(), self.lr)

        # Define learning rate scheduler
        self.scheduler = ReduceLROnPlateau(
            optimizer=self.opt,
            mode="max",
            factor=self.scheduler_factor,
            patience=self.scheduler_patience,
            min_lr=1e-6,
        )
        # Define loss function
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self):
        """
        This function performs model training with the configuration
        specified by the class initialization.
        """
        if not self.only_do_testing:

            # Initialize best accuracy
            if self.use_pretrained_model:
                logging.info("\n------ Using pretrained model ------\n")
                best_epoch, best_acc = self.valid_one_epoch(self.start_epoch, 0, 0)
            else:
                best_epoch, best_acc = 0, 0

            # Loop over epochs (training + validation)
            logging.info("\n------ Begin training ------\n")

            for e in range(best_epoch + 1, best_epoch + self.nb_epochs + 1):
                self.train_one_epoch(e)
                best_epoch, best_acc = self.valid_one_epoch(e, best_epoch, best_acc)

            logging.info(f"\nBest valid acc at epoch {best_epoch}: {best_acc}\n")
            logging.info("\n------ Training finished ------\n")

            # Loading best model
            if self.save_best:
                self.net = torch.load(
                    f"{self.checkpoint_dir}/best_model.pth", map_location=self.device
                )
                logging.info(
                    f"Loading best model, epoch={best_epoch}, valid acc={best_acc}"
                )
            else:
                logging.info(
                    "Cannot load best model because save_best option is "
                    "disabled. Model from last epoch is used for testing."
                )

        # Test trained model
        if self.dataset_name in ["sc", "ssc"]:
            self.test_one_epoch(self.test_loader)
        else:
            self.test_one_epoch(self.valid_loader)
            logging.info(
                "\nThis dataset uses the same split for validation and testing.\n"
            )

    def init_exp_folders(self):
        """
        This function defines the output folders for the experiment.
        """
        # Check if path exists for loading pretrained model
        if self.use_pretrained_model:
            exp_folder = self.load_exp_folder
            self.load_path = exp_folder + "/checkpoints/best_model.pth"
            if not os.path.exists(self.load_path):
                raise FileNotFoundError(
                    errno.ENOENT, os.strerror(errno.ENOENT), self.load_path
                )

        # Use given path for new model folder
        elif self.new_exp_folder is not None:
            exp_folder = self.new_exp_folder

        # Generate a path for new model from chosen config
        else:
            outname = self.dataset_name + "_" + self.model_type + "_"
            outname += str(self.nb_layers) + "lay" + str(self.nb_hiddens)
            outname += "_drop" + str(self.pdrop) + "_" + str(self.normalization)
            outname += "_bias" if self.use_bias else "_nobias"
            outname += "_bdir" if self.bidirectional else "_udir"
            outname += "_reg" if self.use_regularizers else "_noreg"
            outname += "_lr" + str(self.lr)
            exp_folder = "exp/test_exps/" + outname.replace(".", "_")

        # For a new model check that out path does not exist
        if not self.use_pretrained_model and os.path.exists(exp_folder):
            raise FileExistsError(errno.EEXIST, os.strerror(errno.EEXIST), exp_folder)

        # Create folders to store experiment
        self.log_dir = exp_folder + "/log/"
        self.checkpoint_dir = exp_folder + "/checkpoints/"
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        self.exp_folder = exp_folder

    def init_logging(self):
        """
        This function sets the experimental log to be written either to
        a dedicated log file, or to the terminal.
        """
        if self.log_tofile:
            logging.FileHandler(
                filename=self.log_dir + "exp.log",
                mode="a",
                encoding=None,
                delay=False,
            )
            logging.basicConfig(
                filename=self.log_dir + "exp.log",
                level=logging.INFO,
                format="%(message)s",
            )
        else:
            logging.basicConfig(
                level=logging.INFO,
                format="%(message)s",
            )

    def init_dataset(self):
        """
        This function prepares dataloaders for the desired dataset.
        """
        # For the spiking datasets
        if self.dataset_name in ["shd", "ssc"]:

            self.nb_inputs = 700
            self.nb_outputs = 20 if self.dataset_name == "shd" else 35

            self.train_loader = load_shd_or_ssc(
                dataset_name=self.dataset_name,
                data_folder=self.data_folder,
                split="train",
                batch_size=self.batch_size,
                nb_steps=100,
                shuffle=True,
            )
            self.valid_loader = load_shd_or_ssc(
                dataset_name=self.dataset_name,
                data_folder=self.data_folder,
                split="valid",
                batch_size=self.batch_size,
                nb_steps=100,
                shuffle=False,
            )
            if self.dataset_name == "ssc":
                self.test_loader = load_shd_or_ssc(
                    dataset_name=self.dataset_name,
                    data_folder=self.data_folder,
                    split="test",
                    batch_size=self.batch_size,
                    nb_steps=100,
                    shuffle=False,
                )
            if self.use_augm:
                logging.warning(
                    "\nWarning: Data augmentation not implemented for SHD and SSC.\n"
                )

        # For the non-spiking datasets
        elif self.dataset_name in ["hd", "sc"]:

            self.nb_inputs = 40
            self.nb_outputs = 20 if self.dataset_name == "hd" else 35

            self.train_loader = load_hd_or_sc(
                dataset_name=self.dataset_name,
                data_folder=self.data_folder,
                split="train",
                batch_size=self.batch_size,
                use_augm=self.use_augm,
                shuffle=True,
            )
            self.valid_loader = load_hd_or_sc(
                dataset_name=self.dataset_name,
                data_folder=self.data_folder,
                split="valid",
                batch_size=self.batch_size,
                use_augm=self.use_augm,
                shuffle=False,
            )
            if self.dataset_name == "sc":
                self.test_loader = load_hd_or_sc(
                    dataset_name=self.dataset_name,
                    data_folder=self.data_folder,
                    split="test",
                    batch_size=self.batch_size,
                    use_augm=self.use_augm,
                    shuffle=False,
                )
            if self.use_augm:
                logging.info("\nData augmentation is used\n")

        else:
            raise ValueError(f"Invalid dataset name {self.dataset_name}")

    def init_model(self):
        """
        This function either loads pretrained model or builds a
        new model (ANN or SNN) depending on chosen config.
        """
        input_shape = (self.batch_size, None, self.nb_inputs)
        layer_sizes = [self.nb_hiddens] * (self.nb_layers - 1) + [self.nb_outputs]

        if self.use_pretrained_model:
            self.net = torch.load(self.load_path, map_location=self.device)
            logging.info(f"\nLoaded model at: {self.load_path}\n {self.net}\n")

        elif self.model_type in ["LIF", "adLIF", "RLIF", "RadLIF"]:

            self.net = SNN(
                input_shape=input_shape,
                layer_sizes=layer_sizes,
                neuron_type=self.model_type,
                dropout=self.pdrop,
                normalization=self.normalization,
                use_bias=self.use_bias,
                bidirectional=self.bidirectional,
                use_readout_layer=True,
            ).to(self.device)

            logging.info(f"\nCreated new spiking model:\n {self.net}\n")

        elif self.model_type in ["MLP", "RNN", "LiGRU", "GRU"]:

            self.net = ANN(
                input_shape=input_shape,
                layer_sizes=layer_sizes,
                ann_type=self.model_type,
                dropout=self.pdrop,
                normalization=self.normalization,
                use_bias=self.use_bias,
                bidirectional=self.bidirectional,
                use_readout_layer=True,
            ).to(self.device)

            logging.info(f"\nCreated new non-spiking model:\n {self.net}\n")

        else:
            raise ValueError(f"Invalid model type {self.model_type}")

        self.nb_params = sum(
            p.numel() for p in self.net.parameters() if p.requires_grad
        )
        logging.info(f"Total number of trainable parameters is {self.nb_params}")

    def train_one_epoch(self, e):
        """
        This function trains the model with a single pass over the
        training split of the dataset.
        """
        start = time.time()
        self.net.train()
        losses, accs = [], []
        epoch_spike_rate = 0

        # Loop over batches from train set
        for step, (x, _, y) in enumerate(self.train_loader):

            # Dataloader uses cpu to allow pin memory
            x = x.to(self.device)
            y = y.to(self.device)

            # Forward pass through network
            output, firing_rates = self.net(x)

            # Compute loss
            loss_val = self.loss_fn(output, y)
            losses.append(loss_val.item())

            # Spike activity
            if self.net.is_snn:
                epoch_spike_rate += torch.mean(firing_rates)

                if self.use_regularizers:
                    reg_quiet = F.relu(self.reg_fmin - firing_rates).sum()
                    reg_burst = F.relu(firing_rates - self.reg_fmax).sum()
                    loss_val += self.reg_factor * (reg_quiet + reg_burst)

            # Backpropagate
            self.opt.zero_grad()
            loss_val.backward()
            self.opt.step()

            # Compute accuracy with labels
            pred = torch.argmax(output, dim=1)
            acc = np.mean((y == pred).detach().cpu().numpy())
            accs.append(acc)

        # Learning rate of whole epoch
        current_lr = self.opt.param_groups[-1]["lr"]
        logging.info(f"Epoch {e}: lr={current_lr}")

        # Train loss of whole epoch
        train_loss = np.mean(losses)
        logging.info(f"Epoch {e}: train loss={train_loss}")

        # Train accuracy of whole epoch
        train_acc = np.mean(accs)
        logging.info(f"Epoch {e}: train acc={train_acc}")

        # Train spike activity of whole epoch
        if self.net.is_snn:
            epoch_spike_rate /= step
            logging.info(f"Epoch {e}: train mean act rate={epoch_spike_rate}")

        end = time.time()
        elapsed = str(timedelta(seconds=end - start))
        logging.info(f"Epoch {e}: train elapsed time={elapsed}")

    def valid_one_epoch(self, e, best_epoch, best_acc):
        """
        This function tests the model with a single pass over the
        validation split of the dataset.
        """
        with torch.no_grad():

            self.net.eval()
            losses, accs = [], []
            epoch_spike_rate = 0

            # Loop over batches from validation set
            for step, (x, _, y) in enumerate(self.valid_loader):

                # Dataloader uses cpu to allow pin memory
                x = x.to(self.device)
                y = y.to(self.device)

                # Forward pass through network
                output, firing_rates = self.net(x)

                # Compute loss
                loss_val = self.loss_fn(output, y)
                losses.append(loss_val.item())

                # Compute accuracy with labels
                pred = torch.argmax(output, dim=1)
                acc = np.mean((y == pred).detach().cpu().numpy())
                accs.append(acc)

                # Spike activity
                if self.net.is_snn:
                    epoch_spike_rate += torch.mean(firing_rates)

            # Validation loss of whole epoch
            valid_loss = np.mean(losses)
            logging.info(f"Epoch {e}: valid loss={valid_loss}")

            # Validation accuracy of whole epoch
            valid_acc = np.mean(accs)
            logging.info(f"Epoch {e}: valid acc={valid_acc}")

            # Validation spike activity of whole epoch
            if self.net.is_snn:
                epoch_spike_rate /= step
                logging.info(f"Epoch {e}: valid mean act rate={epoch_spike_rate}")

            # Update learning rate
            self.scheduler.step(valid_acc)

            # Update best epoch and accuracy
            if valid_acc > best_acc:
                best_acc = valid_acc
                best_epoch = e

                # Save best model
                if self.save_best:
                    torch.save(self.net, f"{self.checkpoint_dir}/best_model.pth")
                    logging.info(f"\nBest model saved with valid acc={valid_acc}")

            logging.info("\n-----------------------------\n")

            return best_epoch, best_acc

    def test_one_epoch(self, test_loader):
        """
        This function tests the model with a single pass over the
        testing split of the dataset.
        """
        with torch.no_grad():

            self.net.eval()
            losses, accs = [], []
            epoch_spike_rate = 0

            logging.info("\n------ Begin Testing ------\n")

            # Loop over batches from test set
            for step, (x, _, y) in enumerate(test_loader):

                # Dataloader uses cpu to allow pin memory
                x = x.to(self.device)
                y = y.to(self.device)

                # Forward pass through network
                output, firing_rates = self.net(x)

                # Compute loss
                loss_val = self.loss_fn(output, y)
                losses.append(loss_val.item())

                # Compute accuracy with labels
                pred = torch.argmax(output, dim=1)
                acc = np.mean((y == pred).detach().cpu().numpy())
                accs.append(acc)

                # Spike activity
                if self.net.is_snn:
                    epoch_spike_rate += torch.mean(firing_rates)

            # Test loss
            test_loss = np.mean(losses)
            logging.info(f"Test loss={test_loss}")

            # Test accuracy
            test_acc = np.mean(accs)
            logging.info(f"Test acc={test_acc}")

            # Test spike activity
            if self.net.is_snn:
                epoch_spike_rate /= step
                logging.info(f"Test mean act rate={epoch_spike_rate}")

            logging.info("\n-----------------------------\n")

def add_model_options(parser):
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["LIF", "adLIF", "RLIF", "RadLIF", "MLP", "RNN", "LiGRU", "GRU"],
        default="LIF",
        help="Type of ANN or SNN model.",
    )
    parser.add_argument(
        "--nb_layers",
        type=int,
        default=3,
        help="Number of layers (including readout layer).",
    )
    parser.add_argument(
        "--nb_hiddens",
        type=int,
        default=128,
        help="Number of neurons in all hidden layers.",
    )
    parser.add_argument(
        "--pdrop",
        type=float,
        default=0.1,
        help="Dropout rate, must be between 0 and 1.",
    )
    parser.add_argument(
        "--normalization",
        type=str,
        default="batchnorm",
        help="Type of normalization, Every string different from batchnorm "
        "and layernorm will result in no normalization.",
    )
    parser.add_argument(
        "--use_bias",
        type=lambda x: bool(strtobool(str(x))),
        default=False,
        help="Whether to include trainable bias with feedforward weights.",
    )
    parser.add_argument(
        "--bidirectional",
        type=lambda x: bool(strtobool(str(x))),
        default=False,
        help="If True, a bidirectional model that scans the sequence in both "
        "directions is used, which doubles the size of feedforward matrices. ",
    )
    return parser

def add_training_options(parser):
    parser.add_argument(
        "--use_pretrained_model",
        type=lambda x: bool(strtobool(str(x))),
        default=False,
        help="Whether to load a pretrained model or to create a new one.",
    )
    parser.add_argument(
        "--only_do_testing",
        type=lambda x: bool(strtobool(str(x))),
        default=False,
        help="If True, will skip training and only perform testing of the "
        "loaded model.",
    )
    parser.add_argument(
        "--load_exp_folder",
        type=str,
        default=None,
        help="Path to experiment folder with a pretrained model to load. Note "
        "that the same path will be used to store the current experiment.",
    )
    parser.add_argument(
        "--new_exp_folder",
        type=str,
        default='./log/test111',
        help="Path to output folder to store experiment.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        choices=["shd", "ssc", "hd", "sc"],
        default="shd",
        help="Dataset name (shd, ssc, hd or sc).",
    )
    parser.add_argument(
        "--data_folder",
        type=str,
        default="./data/raw/",
        help="Path to dataset folder.",
    )
    parser.add_argument(
        "--log_tofile",
        type=lambda x: bool(strtobool(str(x))),
        default=True,
        help="Whether to print experiment log in an dedicated file or "
        "directly inside the terminal.",
    )
    parser.add_argument(
        "--save_best",
        type=lambda x: bool(strtobool(str(x))),
        default=True,
        help="If True, the model from the epoch with the highest validation "
        "accuracy is saved, if False, no model is saved.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Number of input examples inside a single batch.",
    )
    parser.add_argument(
        "--nb_epochs",
        type=int,
        default=5,
        help="Number of training epochs (i.e. passes through the dataset).",
    )
    parser.add_argument(
        "--start_epoch",
        type=int,
        default=0,
        help="Epoch number to start training at. Will be 0 if no pretrained "
        "model is given. First epoch will be start_epoch+1.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-2,
        help="Initial learning rate for training. The default value of 0.01 "
        "is good for SHD and SC, but 0.001 seemed to work better for HD and SC.",
    )
    parser.add_argument(
        "--scheduler_patience",
        type=int,
        default=1,
        help="Number of epochs without progress before the learning rate "
        "gets decreased.",
    )
    parser.add_argument(
        "--scheduler_factor",
        type=float,
        default=0.7,
        help="Factor between 0 and 1 by which the learning rate gets "
        "decreased when the scheduler patience is reached.",
    )
    parser.add_argument(
        "--use_regularizers",
        type=lambda x: bool(strtobool(str(x))),
        default=False,
        help="Whether to use regularizers in order to constrain the "
        "firing rates of spiking neurons within a given range.",
    )
    parser.add_argument(
        "--reg_factor",
        type=float,
        default=0.5,
        help="Factor that scales the loss value from the regularizers.",
    )
    parser.add_argument(
        "--reg_fmin",
        type=float,
        default=0.01,
        help="Lowest firing frequency value of spiking neurons for which "
        "there is no regularization loss.",
    )
    parser.add_argument(
        "--reg_fmax",
        type=float,
        default=0.5,
        help="Highest firing frequency value of spiking neurons for which "
        "there is no regularization loss.",
    )
    parser.add_argument(
        "--use_augm",
        type=lambda x: bool(strtobool(str(x))),
        default=False,
        help="Whether to use data augmentation or not. Only implemented for "
        "nonspiking HD and SC datasets.",
    )
    return parser

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Model training on spiking speech commands datasets."
    )
    parser = add_model_options(parser)
    parser = add_training_options(parser)
    args = parser.parse_args()
    experiment = Experiment(args)

    # Run experiment
    experiment.forward()