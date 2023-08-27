'''
首修2023年7月26日20:30:22
大修2023年8月4日17:57:38
修改2023年8月18日15:45:30
https://github.com/idiap/sparch/blob/main/sparch/models/snns.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from distutils.util import strtobool
import numpy as np
import matplotlib.pyplot as plt
import time, warnings, errno, os, h5py, logging, argparse
warnings.filterwarnings("ignore")
from datetime import timedelta
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
logger = logging.getLogger(__name__)


def add_training_options(parser):
    date = time.strftime("%Y-%m-%d-%H-%M-%S/", time.localtime(time.time()))[5:16]
    parser.add_argument("--use_pretrained_model", type=bool,   default=False,            help="Whether to load a pretrained model or to create a new one.",)
    parser.add_argument("--only_do_testing",      type=bool,   default=False,            help="If True, will skip training and only perform testing of the loaded model.",)
    parser.add_argument("--load_exp_folder",      type=str,    default=None,             help="Path to experiment folder with a pretrained model to load. Note that the same path will be used to store the current experiment.",)
    parser.add_argument("--new_exp_folder",       type=str,    default='./log/' + date,  help="Path to output folder to store experiment.",)
    parser.add_argument("--dataset_name",         type=str,    default='shd',            help="Dataset name (shd, ssc, hd or sc).",)
    parser.add_argument("--data_folder",          type=str,    default="./data/raw/",    help="Path to dataset folder.",)
    parser.add_argument("--log_tofile",           type=bool,   default=True,             help="Whether to print experiment log in an dedicated file or directly inside the terminal.",)
    parser.add_argument("--save_best",            type=bool,   default=True,             help="If True, the model from the epoch with the highest validation accuracy is saved, if False, no model is saved.",)
    parser.add_argument("--batch_size",           type=int,    default=512,              help="Number of input examples inside a single batch.",)
    parser.add_argument("--nb_epochs",            type=int,    default=50,               help="Number of training epochs (i.e. passes through the dataset).",)
    parser.add_argument("--start_epoch",          type=int,    default=0,                help="Epoch number to start training at. Will be 0 if no pretrained model is given. First epoch will be start_epoch+1.",)
    parser.add_argument("--lr",                   type=float,  default=1.5e-2,             help="Initial learning rate for training. The default value of 0.01 is good for SHD and SC, but 0.001 seemed to work better for HD and SC.",)
    parser.add_argument("--scheduler_patience",   type=int,    default=1,                help="Number of epochs without progress before the learning rate gets decreased.",)
    parser.add_argument("--scheduler_factor",     type=float,  default=0.7,              help="Factor between 0 and 1 by which the learning rate gets decreased when the scheduler patience is reached.",)
    parser.add_argument("--use_regularizers",     type=bool,   default=True,             help="Whether to use regularizers in order to constrain the firing rates of spiking neurons within a given range.",)
    parser.add_argument("--reg_factor",           type=float,  default=0.5,              help="Factor that scales the loss value from the regularizers.",)
    parser.add_argument("--reg_fmin",             type=float,  default=0.01,             help="Lowest firing frequency value of spiking neurons for which there is no regularization loss.",)
    parser.add_argument("--reg_fmax",             type=float,  default=0.2,              help="Highest firing frequency value of spiking neurons for which there is no regularization loss.",)
    parser.add_argument("--use_augm",             type=bool,   default=False,            help="Whether to use data augmentation or not. Only implemented for nonspiking HD and SC datasets.",)
    parser.add_argument("--nb_steps",             type=int,    default=50,)
    parser.add_argument("--trial",                type=int,    default=5,)
    parser.add_argument("--seed",                 type=int,    default=round(time.time()),) # round(time.time())
    parser.add_argument("--ckpt_freq",            type=int,    default=5,)
    parser.add_argument("--threshold",            type=float,  default=1.0,)
    parser.add_argument("--date",                 type=str,    default=date,)
    
    parser.add_argument("--model_type",           type=str,    default="RadLIF",    help="Type of ANN or SNN model.",)
    parser.add_argument("--nb_layers",            type=int,    default=3,           help="Number of layers (including readout layer).",)
    parser.add_argument("--nb_hiddens",           type=int,    default=1024,        help="Number of neurons in all hidden layers.",)
    parser.add_argument("--pdrop",                type=float,  default=0.1,         help="Dropout rate, must be between 0 and 1.",)
    parser.add_argument("--normalization",        type=str,    default="batchnorm", help="Type of normalization, Every string different from batchnorm and layernorm will result in no normalization.",)
    parser.add_argument("--use_bias",             type=bool,   default=True,        help="Whether to include trainable bias with feedforward weights.",)
    parser.add_argument("--bidirectional",        type=bool,   default=False,       help="If True, a bidirectional model that scans the sequence in both directions is used, which doubles the size of feedforward matrices. ",)
    parser.add_argument("--train_input",          type=bool,   default=True,)
    parser.add_argument("--dropout",              type=float,  default=0.0,)
    parser.add_argument("--dropout_stop",         type=float,  default=0.95,)
    parser.add_argument("--dropout_stepping",     type=float,  default=0.0,)
    parser.add_argument("--clustering",           type=bool,   default=False,)
    parser.add_argument("--clustering_factor",    type=list,   default=[1, 2.5],)
    parser.add_argument("--cin_minmax",           type=list,   default=[0.01, 0.05],)
    parser.add_argument("--cout_minmax",          type=list,   default=[0.05, 0.2],)
    parser.add_argument("--nb_cluster",           type=int,    default=8,)
    parser.add_argument("--noise_test",           type=float,  default=0.0,)
    return parser

def print_options(args):
    logging.info(
        """
        Training Config
        ---------------
        Date: {date}
        Use pretrained model: {use_pretrained_model}
        Only do testing: {only_do_testing}
        Load experiment folder: {load_exp_folder}
        New experiment folder: {new_exp_folder}
        Dataset name: {dataset_name}
        Data folder: {data_folder}
        Log to file: {log_tofile}
        Save best model: {save_best}
        Batch size: {batch_size}
        Number of epochs: {nb_epochs}
        Start epoch: {start_epoch}
        Initial learning rate: {lr}
        Scheduler patience: {scheduler_patience}
        Scheduler factor: {scheduler_factor}
        Use regularizers: {use_regularizers}
        Regularization factor: {reg_factor}
        Regularization min firing rate: {reg_fmin}
        Reguarization max firing rate: {reg_fmax}
        Use data augmentation: {use_augm}
        Number of steps: {nb_steps}
        Trials: {trial}
        Seed: {seed}
        Checkpoint frequency: {ckpt_freq}
        Threshold: {threshold}
        
        ---------------
        Model Config
        
        Model Type: {model_type}
        Number of layers: {nb_layers}
        Number of hidden neurons: {nb_hiddens}
        Dropout rate: {pdrop}
        Normalization: {normalization}
        Use bias: {use_bias}
        Bidirectional: {bidirectional}
        Train input layer: {train_input}
        Dropout: {dropout}
        Dropout_stop: {dropout_stop}
        Dropout_stepping: {dropout_stepping}
        Clustering: {clustering}
        Clustering factor: {clustering_factor}
        Cin min and max: {cin_minmax}
        Cout min and max: {cout_minmax}
        Number of clusters: {nb_cluster}
        Noise in testset: {noise_test}
    """.format(**vars(args))
    )


##########################################################
########### define surrogate gradient function ###########
class SpikeFunctionBoxcar(torch.autograd.Function):
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

#######################################
########### define RC model ###########
class SNN(nn.Module):
    """
    A multi-layered Spiking Neural Network (SNN).
    It accepts input tensors formatted as (batch, time, feat). 
    The function returns the outputs of the last spiking or readout layer with shape (batch, time, feats) or (batch, feats) respectively, as well as the firing rates of all hidden neurons with shape (num_layers*feats).
    """
    def __init__(self, input_shape, layer_sizes, neuron_type, threshold, pdrop, normalization, use_bias, bidirectional, use_readout_layer, train_input, dropout):
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
        self.pdrop = pdrop
        self.normalization = normalization
        self.use_bias = use_bias
        self.bidirectional = bidirectional
        self.use_readout_layer = use_readout_layer
        self.train_input = train_input
        self.dropout = dropout
        self.snn = self._init_layers()

    def _init_layers(self):
        snn = nn.ModuleList([])
        input_size = self.input_size
        snn_class = self.neuron_type + "Layer"
        if self.use_readout_layer: num_hidden_layers = self.num_layers - 1
        else:                      num_hidden_layers = self.num_layers
        for i in range(num_hidden_layers):
            snn.append(
                globals()[snn_class](
                    input_size=input_size,
                    hidden_size=self.layer_sizes[i],
                    batch_size=self.batch_size,
                    threshold=self.threshold,
                    pdrop=self.pdrop,
                    normalization=self.normalization,
                    use_bias=self.use_bias,
                    bidirectional=self.bidirectional,
                    train_input=self.train_input,
                    dropout=self.dropout,
                )
            )
            input_size = self.layer_sizes[i] * (1 + self.bidirectional)

        if self.use_readout_layer:
            snn.append(
                ReadoutLayer(
                    input_size=input_size,
                    hidden_size=self.layer_sizes[-1],
                    batch_size=self.batch_size,
                    pdrop=self.pdrop,
                    normalization=self.normalization,
                    use_bias=self.use_bias,
                )
            )
        return snn

    def forward(self, x, mask):
        # Reshape input tensors to (batch, time, feats) for 4d inputs
        if self.reshape:
            if x.ndim == 4:
                x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])
            else:
                raise NotImplementedError

        all_spikes = []
        for i, snn_lay in enumerate(self.snn):
            x = snn_lay(x, mask[i])
            if not (self.use_readout_layer and i == self.num_layers - 1):
                all_spikes.append(x)

        firing_rates = torch.cat(all_spikes, dim=2).mean(dim=(0, 1)) # Compute mean firing rate of each spiking neuron
        return x, firing_rates, all_spikes

class RadLIFLayer(nn.Module):
    """A single layer of adaptive Leaky Integrate-and-Fire neurons with layer-wise recurrent connections (RadLIF)."""
    def __init__(self, input_size, hidden_size, batch_size, threshold, pdrop, normalization, use_bias, bidirectional, train_input, dropout):
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.batch_size = batch_size
        self.threshold = threshold
        self.normalization = normalization
        self.use_bias = use_bias
        self.bidirectional = bidirectional
        self.train_input = train_input
        self.dropout = dropout
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
        self.drop = nn.Dropout(p=pdrop)
        
        if not self.train_input:
            for name, p in self.named_parameters():
                if 'W' in name: p.requires_grad = False

    def forward(self, x, mask):
        # x.shape = [batch, nb_steps, input]
        # Wx.shape = [batch, nb_steps, hid]
        if self.bidirectional: # Concatenate flipped sequence on batch dim
            x_flip = x.flip(1)
            x = torch.cat([x, x_flip], dim=0)
        if self.batch_size != x.shape[0]: self.batch_size = x.shape[0]
        Wx = self.W(x)
        if self.normalize:
            _Wx = self.norm(Wx.reshape(Wx.shape[0] * Wx.shape[1], Wx.shape[2]))
            Wx = _Wx.reshape(Wx.shape[0], Wx.shape[1], Wx.shape[2])
        s = self.mem_update(Wx, mask) # s.shape=[batch, nb_steps, hid]
        if self.bidirectional: # Concatenate forward and backward sequences on feat dim
            s_f, s_b = s.chunk(2, dim=0)
            s_b = s_b.flip(1)
            s = torch.cat([s_f, s_b], dim=2)
        s = self.drop(s)
        return s

    def mem_update(self, Wx, mask):
        ut = torch.rand(Wx.shape[0], Wx.shape[2]).to(Wx.device)
        wt = torch.rand(Wx.shape[0], Wx.shape[2]).to(Wx.device)
        st = torch.rand(Wx.shape[0], Wx.shape[2]).to(Wx.device)
        s = []
        # Bound values of the neuron parameters to plausible ranges
        alpha = torch.clamp(self.alpha, min=self.alpha_lim[0], max=self.alpha_lim[1])
        beta = torch.clamp(self.beta, min=self.beta_lim[0], max=self.beta_lim[1])
        a = torch.clamp(self.a, min=self.a_lim[0], max=self.a_lim[1])
        b = torch.clamp(self.b, min=self.b_lim[0], max=self.b_lim[1])
        if self.dropout > 0: self.V.weight.data = self.V.weight.data * mask.T
        V = self.V.weight.clone().fill_diagonal_(0)
        for t in range(Wx.shape[1]):
            wt = beta * wt + a * ut + b * st
            ut = alpha * (ut - st) + (1 - alpha) * (Wx[:, t, :] + torch.matmul(st, V) - wt)
            st = self.spike_fct(ut - self.threshold)
            s.append(st)
        return torch.stack(s, dim=1)

class ReadoutLayer(nn.Module):
    """
    This function implements a single layer of non-spiking Leaky Integrate and
    Fire (LIF) neurons, where the output consists of a cumulative sum of the
    membrane potential using a softmax function, instead of spikes.
    """

    def __init__(self, input_size, hidden_size, batch_size, pdrop, normalization, use_bias,):
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.batch_size = batch_size
        self.normalization = normalization
        self.use_bias = use_bias
        self.alpha_lim = [np.exp(-1 / 5), np.exp(-1 / 25)]

        self.W = nn.Linear(self.input_size, self.hidden_size, bias=use_bias)
        self.alpha = nn.Parameter(torch.Tensor(self.hidden_size))
        nn.init.uniform_(self.alpha, self.alpha_lim[0], self.alpha_lim[1])

        self.normalize = False
        if normalization == "batchnorm":
            self.norm = nn.BatchNorm1d(self.hidden_size, momentum=0.05)
            self.normalize = True
        elif normalization == "layernorm":
            self.norm = nn.LayerNorm(self.hidden_size)
            self.normalize = True
        self.drop = nn.Dropout(p=pdrop)

    def forward(self, x, mask):
        Wx = self.W(x) # Feed-forward affine transformations (all steps in parallel)
        if self.normalize:
            _Wx = self.norm(Wx.reshape(Wx.shape[0] * Wx.shape[1], Wx.shape[2]))
            Wx = _Wx.reshape(Wx.shape[0], Wx.shape[1], Wx.shape[2])
        out = self.mem_update(Wx) # Wx.shape=[batch, nb_steps, output], out.shape=[batch, output]
        return out

    def mem_update(self, Wx):
        ut = torch.rand(Wx.shape[0], Wx.shape[2]).to(Wx.device)
        out = torch.zeros(Wx.shape[0], Wx.shape[2]).to(Wx.device)
        alpha = torch.clamp(self.alpha, min=self.alpha_lim[0], max=self.alpha_lim[1]) # Bound values of the neuron parameters to plausible ranges
        for t in range(Wx.shape[1]):
            ut = alpha * ut + (1 - alpha) * Wx[:, t, :]
            out = out + F.softmax(ut, dim=1)
        return out

class SpikingDataset(Dataset):
    def __init__(self, dataset_name, data_folder, split, nb_steps,):
        # Fixed parameters
        self.device = "cpu"  # to allow pin memory
        self.nb_steps = nb_steps
        self.nb_units = 700
        self.max_time = 1.4
        self.time_bins = np.linspace(0, self.max_time, num=self.nb_steps)

        filename = f"{data_folder}/{dataset_name}_{split}.h5"
        self.h5py_file = h5py.File(filename, "r")
        self.firing_times = self.h5py_file["spikes"]["times"]
        self.units_fired = self.h5py_file["spikes"]["units"]
        self.labels = np.array(self.h5py_file["labels"], dtype=int)

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

def load_shd_or_ssc(dataset_name, data_folder, split, batch_size, nb_steps=100, shuffle=True, workers=0,):
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

def init_mask(nb_hiddens, nb_cluster):
    nb_per_cluster = int(args.nb_hiddens / args.nb_cluster)
    a = torch.zeros((args.nb_hiddens, args.nb_hiddens), dtype=torch.int)
    for i in range(args.nb_cluster): 
        a[i*nb_per_cluster:(i+1)*nb_per_cluster, i*nb_per_cluster:(i+1)*nb_per_cluster] = 1.
    invalid_zeros = 1-(a==1).sum().item()/args.nb_hiddens**2
    if invalid_zeros < args.dropout:
        b = (torch.rand(args.nb_hiddens, args.nb_hiddens) > (args.dropout-invalid_zeros)/(1-invalid_zeros)).int() * (1-torch.eye(args.nb_hiddens, args.nb_hiddens, dtype=int))
        mask = a & b
        mask += torch.eye(args.nb_hiddens, args.nb_hiddens, dtype=int)
    else: mask = a
    return mask

class Experiment:
    def __init__(self, args):
        # New model config
        self.model_type = args.model_type
        self.nb_layers = args.nb_layers
        self.nb_hiddens = args.nb_hiddens
        self.pdrop = args.pdrop
        self.normalization = args.normalization
        self.use_bias = args.use_bias
        self.bidirectional = args.bidirectional
        self.threshold = args.threshold

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
        self.nb_steps = args.nb_steps
        self.train_input = args.train_input
        self.noise_test = args.noise_test
        self.seed = args.seed
        self.dropout = args.dropout
        self.nb_cluster = args.nb_cluster
        self.nb_per_cluster = int(args.nb_hiddens / args.nb_cluster)

        self.set_seed(self.seed)
        self.init_exp_folders()
        self.init_logging()
        print_options(args)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"\nDevice is set to {self.device}\n")
        self.init_dataset()
        self.init_model()
        logging.info(f"\nCreated new spiking model:\n {self.net}\n")
        self.nb_params = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
        logging.info(f"Total number of trainable parameters is {self.nb_params}")

        self.optimizer = torch.optim.Adam(self.net.parameters(), self.lr)
        self.scheduler = ReduceLROnPlateau(
            optimizer=self.optimizer,
            mode="max",
            factor=self.scheduler_factor,
            patience=self.scheduler_patience,
            min_lr=1e-6,
        )
        self.loss_fn = nn.CrossEntropyLoss()

    def set_seed(self, seed):
        if seed > 0:
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def forward(self, trial):
        if not self.only_do_testing:
            self.init_model()
            train_accs, valid_accs = [], []
            if self.use_pretrained_model:
                logging.info("\n------ Using pretrained model ------\n")
                best_epoch, best_acc = self.valid_one_epoch(self.start_epoch, 0, 0)
            else:
                best_epoch, best_acc = 0, 0

            logging.info("\n------ Begin training ------\n")
            m1 = init_mask(self.nb_hiddens, self.nb_cluster); m2 = init_mask(self.nb_hiddens, self.nb_cluster)
            # m1 = (torch.rand(config.hid, config.hid) > config.dropout).int() * (1-torch.eye(config.hid, config.hid)).int()
            # m2 = (torch.rand(config.hid, config.hid) > config.dropout).int() * (1-torch.eye(config.hid, config.hid)).int()
            mask = [m1.float().to(self.device), m2.float().to(self.device), 0]
            for e in range(best_epoch + 1, best_epoch + self.nb_epochs + 1):
                train_acc = self.train_one_epoch(e, mask); train_accs.append(train_acc)
                best_epoch, best_acc = self.valid_one_epoch(trial, e, mask, best_epoch, best_acc); valid_accs.append(best_acc)

            logging.info(f"\nBest valid acc at epoch {best_epoch}: {best_acc}\n")
            logging.info("\n------ Training finished ------\n")
            
            self.net = torch.load(f"{self.checkpoint_dir}/best_model_{trial}_{best_acc}.pth", map_location=self.device)
            logging.info(f"Loading best model, epoch={best_epoch}, valid acc={best_acc}")

        # Test trained model
        if self.dataset_name == "ssc": self.test_one_epoch(self.test_loader)
        else:
            self.test_one_epoch(self.valid_loader)
            logging.info("\nThis dataset uses the same split for validation and testing.\n")
        return np.array(train_accs), np.array(valid_accs)

    def init_exp_folders(self):
        """Define the output folders for the experiment."""
        # Check if path exists for loading pretrained model
        if self.use_pretrained_model:
            exp_folder = self.load_exp_folder
            self.load_path = exp_folder + "/checkpoints/best_model.pth"
            if not os.path.exists(self.load_path):
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), self.load_path)

        # Use given path for new model folder
        elif self.new_exp_folder is not None: exp_folder = self.new_exp_folder

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
        if not os.path.exists(self.log_dir):         os.makedirs(self.log_dir)
        if not os.path.exists(self.checkpoint_dir):  os.makedirs(self.checkpoint_dir)
        self.exp_folder = exp_folder

    def init_logging(self):
        if self.log_tofile:
            logging.FileHandler(filename=self.log_dir + "exp.log", mode="a", encoding=None, delay=False,)
            logging.basicConfig(filename=self.log_dir + "exp.log", level=logging.INFO, format="%(message)s",)
        else:
            logging.basicConfig(level=logging.INFO, format="%(message)s",)

    def init_dataset(self):
        if self.dataset_name in ["shd", "ssc"]:
            self.nb_inputs = 700
            self.nb_outputs = 20 if self.dataset_name == "shd" else 35

            self.train_loader = load_shd_or_ssc(dataset_name=self.dataset_name, data_folder=self.data_folder, split="train", batch_size=self.batch_size, nb_steps=self.nb_steps, shuffle=True,)
            self.valid_loader = load_shd_or_ssc(dataset_name=self.dataset_name, data_folder=self.data_folder, split="valid", batch_size=self.batch_size, nb_steps=self.nb_steps, shuffle=False,)
            if self.dataset_name == "ssc":
                self.test_loader = load_shd_or_ssc(dataset_name=self.dataset_name, data_folder=self.data_folder, split="test", batch_size=self.batch_size, nb_steps=self.nb_steps, shuffle=False,)
            if self.use_augm:
                logging.warning("\nWarning: Data augmentation not implemented for SHD and SSC.\n")

    def init_model(self):
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
                threshold=self.threshold,
                pdrop=self.pdrop,
                normalization=self.normalization,
                use_bias=self.use_bias,
                bidirectional=self.bidirectional,
                use_readout_layer=True,
                train_input=self.train_input,
                dropout=self.dropout,
            ).to(self.device)

    def train_one_epoch(self, e, mask):
        start = time.time()
        self.net.train()
        losses, accs, epoch_spike_rate = [], [], 0

        for step, (x, _, y) in enumerate(self.train_loader):
            x = x.to(self.device); y = y.to(self.device)
            output, firing_rates, all_spikes = self.net(x, mask)
            loss_val = self.loss_fn(output, y)

            epoch_spike_rate += torch.mean(firing_rates)
            if self.use_regularizers:
                reg_quiet = F.relu(self.reg_fmin - firing_rates).sum()
                reg_burst = F.relu(firing_rates - self.reg_fmax).sum()
                loss_val += self.reg_factor * (reg_quiet + reg_burst)

            losses.append(loss_val.item())
            self.optimizer.zero_grad()
            loss_val.backward()
            self.optimizer.step()

            pred = torch.argmax(output, dim=1)
            acc = np.mean((y == pred).detach().cpu().numpy())
            accs.append(acc)

        current_lr = self.optimizer.param_groups[-1]["lr"]
        train_loss = np.mean(losses)
        train_acc = np.mean(accs)
        epoch_spike_rate /= step
        elapsed = str(timedelta(seconds=time.time() - start))[5:]
        logging.info(f"Epoch {e}: train loss={train_loss:.4f}, acc={train_acc:.4f}, fr={epoch_spike_rate:.4f}, lr={current_lr:.4f}, time={elapsed}")
        return train_acc

    def valid_one_epoch(self, trial, e, mask, best_epoch, best_acc):
        start = time.time()
        with torch.no_grad():
            self.net.eval()
            losses, accs, epoch_spike_rate = [], [], 0
            for step, (x, _, y) in enumerate(self.valid_loader):
                x += torch.rand_like(x) * self.noise_test
                x = x.to(self.device); y = y.to(self.device)
                output, firing_rates, all_spikes = self.net(x, mask)
                loss_val = self.loss_fn(output, y)
                losses.append(loss_val.item())

                pred = torch.argmax(output, dim=1)
                accs.append(np.mean((y == pred).detach().cpu().numpy()))
                epoch_spike_rate += torch.mean(firing_rates)

            valid_loss = np.mean(losses); valid_acc = np.mean(accs)
            epoch_spike_rate /= step
            elapsed = str(timedelta(seconds=time.time() - start))[5:]
            sparsity = ((mask[0]==0).sum().item()/self.nb_hiddens**2 + (mask[1]==0).sum().item()/self.nb_hiddens**2)/2
            logging.info(f"Epoch {e}: valid loss={valid_loss:.4f}, acc={valid_acc:.4f}, fr={epoch_spike_rate:.4f}, mask={sparsity:.4f}, time={elapsed}")
            self.scheduler.step(valid_acc)

            if valid_acc > best_acc:
                best_acc = valid_acc; best_epoch = e
                torch.save(self.net, f"{self.checkpoint_dir}/best_model_{trial}_{valid_acc}.pth")
                logging.info(f"\nBest model saved with valid acc={valid_acc}")

            logging.info("\n-----------------------------\n")
            return best_epoch, best_acc

    def test_one_epoch(self, test_loader):
        with torch.no_grad():
            self.net.eval()
            losses, accs, epoch_spike_rate = [], [], 0
            logging.info("\n------ Begin Testing ------\n")
            for step, (x, _, y) in enumerate(test_loader):
                x += torch.rand_like(x) * self.noise_test
                x = x.to(self.device); y = y.to(self.device)
                output, firing_rates, all_spikes = self.net(x, [0,0,0])
                loss_val = self.loss_fn(output, y)
                losses.append(loss_val.item())

                pred = torch.argmax(output, dim=1)
                acc = np.mean((y == pred).detach().cpu().numpy())
                accs.append(acc)
                epoch_spike_rate += torch.mean(firing_rates)

            test_loss = np.mean(losses); test_acc = np.mean(accs)
            epoch_spike_rate /= step
            logging.info(f"Test loss={test_loss}, acc={test_acc}, mean act rate={epoch_spike_rate}")
            logging.info("\n-----------------------------\n")

def plot_errorbar(args, train_acc_log, test_acc_log, file_name):
    train_mean = np.mean(train_acc_log, axis=1)
    train_std = np.std(train_acc_log, axis=1)
    # train_var = np.var(train_acc_log, axis=1)
    # train_max = np.max(train_acc_log, axis=1)
    # train_min = np.min(train_acc_log, axis=1)

    test_mean = np.mean(test_acc_log, axis=1)
    test_std = np.std(test_acc_log, axis=1)
    # test_var = np.var(test_acc_log, axis=1)
    # test_max = np.max(test_acc_log, axis=1)
    # test_min = np.min(test_acc_log, axis=1)

    plt.plot(list(range(args.nb_epochs)), train_mean, color='deeppink', label='train')
    plt.fill_between(list(range(args.nb_epochs)), train_mean-train_std, train_mean+train_std, color='deeppink', alpha=0.2)
    # plt.fill_between(list(range(args.epoch)), train_min, train_max, color='violet', alpha=0.2)

    plt.plot(list(range(args.nb_epochs)), test_mean, color='blue', label='test')
    plt.fill_between(list(range(args.nb_epochs)), test_mean-test_std, test_mean+test_std, color='blue', alpha=0.2)
    # plt.fill_between(list(range(args.epoch)), test_min, test_max, color='blue', alpha=0.2)

    plt.legend()
    plt.grid()
    # plt.axis([-5, 105, 75, 95])
    plt.savefig(file_name)
    # plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model training on spiking speech commands datasets.")
    parser = add_training_options(parser)
    args = parser.parse_args()
    experiment = Experiment(args)

    log = np.zeros((2, args.nb_epochs, args.trial))
    for i in range(args.trial):
        logging.info(f"\n---------------Trial:{i+1}---------------\n")
        experiment.set_seed(args.seed + i + 1)
        train_accs, valid_accs = experiment.forward(i+1)
        log[0,:,i] = train_accs
        log[1,:,i] = valid_accs
    plot_errorbar(args, log[0], log[1], './fig/'+ args.date +'.pdf')