'''
首修2023年7月26日20:30:22

'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import time, warnings, errno, os, h5py, logging
warnings.filterwarnings("ignore")
from datetime import timedelta
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
logger = logging.getLogger(__name__)

class config:
    neuron_type = 'RadLIF'      # ["LIF", "adLIF", "RLIF", "RadLIF", "MLP", "RNN", "LiGRU", "GRU"]
    nb_inputs = 700
    nb_outputs = 20
    nb_layers = 3               # Number of layers (including readout layer).
    nb_hiddens = 1024           # Number of neurons in all hidden layers.
    nb_steps = 100
    pdrop = 0.1                 # Dropout rate, must be between 0 and 1.
    normalization = "batchnorm" # Type of normalization, Every string different from batchnorm and layernorm will result in no normalization.
    use_bias = False            # Whether to include trainable bias with feed-forward weights.
    bidirectional = False       # If True, a bidirectional model that scans the sequence in both directions is used, which doubles the size of feed-forward matrices in layers l>0.
    
    date = time.strftime("%Y-%m-%d-%H-%M-%S/", time.localtime(time.time()))[5:16]
    use_pretrained_model = False
    only_do_testing = False
    load_exp_folder = None      # Path to experiment folder with a pretrained model to load. Note that the same path will be used to store the current experiment.
    new_exp_folder = './log/' + date      # Path to output folder to store experiment.
    dataset_name = 'shd'        # ["shd", "ssc"]
    data_folder = "./data/raw/"
    batch_size = 256
    nb_epochs = 30
    
    train_input_layer = True
    trial = 5
    seed = -100 # round(time.time())
    dropout = 0
    dropout_stop = 0.95
    dropout_stepping = 0.0
    ckpt_freq = 5
    clustering = False
    clustering_factor = [1, 2.5]
    cin_minmax = [0.001, 0.05]
    cout_minmax = [0.05, 0.2]
    nb_cluster = 8
    nb_per_cluster = int(nb_hiddens/nb_cluster)
    noise_test = 0.0            # add rand(0,1) noise to test dataset with given noise strength.
    
    
    start_epoch = 0             # Epoch number to start training at. Will be 0 if no pretrained model is given. First epoch will be start_epoch+1.
    lr = 1e-2                   # Initial learning rate for training. The default value of 0.01 is good for SHD and SC, but 0.001 seemed to work better for HD and SC.
    scheduler_patience = 2      # Number of epochs without progress before the learning rate gets decreased.
    scheduler_factor = 0.7      # Factor between 0 and 1 by which the learning rate gets decreased when the scheduler patience is reached.
    use_regularizers = True     # Whether to use regularizers in order to constrain the firing rates of spiking neurons within a given range.
    reg_factor = 0.5            # Factor that scales the loss value from the regularizers.
    reg_fmin = 0.01             # Lowest firing frequency value of spiking neurons for which there is no regularization loss.
    reg_fmax = 0.1              # Highest firing frequency value of spiking neurons for which there is no regularization loss.
    use_augm = False            # Whether to use data augmentation or not. Only implemented for non-spiking HD and SC datasets.
    use_readout_layer = True    # If True, the final layer is a non-spiking, non-recurrent LIF and outputs a cumulative sum of the membrane potential over time. The outputs have shape (batch, labels) with no time dimension. If False, the final layer is the same as the hidden layers and outputs spike trains with shape (batch, time, labels).
    threshold = 1.0             # Fixed threshold value for the membrane potential.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    The function returns the outputs of the last spiking or readout layer
    with shape (batch, time, feats) or (batch, feats) respectively, as well
    as the firing rates of all hidden neurons with shape (num_layers*feats).
    """
    def __init__(self):
        super(SNN, self).__init__()
        # Fixed parameters
        self.input_shape = (config.batch_size, config.nb_steps, config.nb_inputs)
        self.layer_sizes = [config.nb_hiddens] * (config.nb_layers - 1) + [config.nb_outputs]
        self.input_size = float(torch.prod(torch.tensor(self.input_shape[2:])))
        self.snn = self._init_layers()

    def _init_layers(self):
        snn = nn.ModuleList([])
        input_size = self.input_size
        snn_class = config.neuron_type + "Layer"

        if config.use_readout_layer: num_hidden_layers = config.nb_layers - 1
        else:                        num_hidden_layers = config.nb_layers

        for i in range(num_hidden_layers):
            snn.append(globals()[snn_class](input_size=input_size, hidden_size=self.layer_sizes[i],))
            input_size = self.layer_sizes[i] * (1 + config.bidirectional)
        if config.use_readout_layer:
            snn.append(ReadoutLayer(input_size=input_size, hidden_size=self.layer_sizes[-1],))
        return snn

    def forward(self, x, mask):
        all_spikes = []
        for i, snn_layer in enumerate(self.snn):
            x = snn_layer(x, mask[i])
            if not (config.use_readout_layer and i == config.nb_layers - 1):
                all_spikes.append(x)
        firing_rates = torch.cat(all_spikes, dim=2).mean(dim=(0, 1)) # Compute mean firing rate of each spiking neuron
        return x, firing_rates, all_spikes

class RadLIFLayer(nn.Module):
    """A single layer of adaptive Leaky Integrate-and-Fire neurons with layer-wise recurrent connections (RadLIF)."""
    def __init__(self, input_size, hidden_size,):
        super().__init__()
        # Fixed parameters
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.alpha_lim = [np.exp(-1 / 5), np.exp(-1 / 25)]
        self.beta_lim = [np.exp(-1 / 30), np.exp(-1 / 120)]
        self.a_lim = [-1.0, 1.0]
        self.b_lim = [0.0, 2.0]
        self.spike_fct = SpikeFunctionBoxcar.apply

        # Trainable parameters
        self.W = nn.Linear(self.input_size, self.hidden_size, bias=config.use_bias)
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

        self.normalize = False
        if config.normalization == "batchnorm":
            self.norm = nn.BatchNorm1d(self.hidden_size, momentum=0.05)
            self.normalize = True
        elif config.normalization == "layernorm":
            self.norm = nn.LayerNorm(self.hidden_size)
            self.normalize = True
        self.drop = nn.Dropout(p=config.pdrop)
        
        # if not config.train_input_layer:
        #     for name, p in self.named_parameters():
        #         if 'W' in name: p.requires_grad = False

    def forward(self, x, mask):
        # x.shape = [batch, nb_steps, input]
        # Wx.shape = [batch, nb_steps, hid]
        # Concatenate flipped sequence on batch dim
        if config.bidirectional:
            x_flip = x.flip(1)
            x = torch.cat([x, x_flip], dim=0)
        Wx = self.W(x) # Feed-forward affine transformations (all steps in parallel)
        if self.normalize:
            _Wx = self.norm(Wx.reshape(Wx.shape[0] * Wx.shape[1], Wx.shape[2]))
            Wx = _Wx.reshape(Wx.shape[0], Wx.shape[1], Wx.shape[2])
        s = self.mem_update(Wx, mask) # s.shape=[batch, nb_steps, hid]
        # Concatenate forward and backward sequences on feat dim
        if config.bidirectional:
            s_f, s_b = s.chunk(2, dim=0)
            s_b = s_b.flip(1)
            s = torch.cat([s_f, s_b], dim=2)
        s = self.drop(s)
        return s

    def mem_update(self, Wx, mask):
        ut = torch.rand(Wx.shape[0], Wx.shape[2]).to(config.device) # [batch, hid]
        wt = torch.rand(Wx.shape[0], Wx.shape[2]).to(config.device)
        st = torch.rand(Wx.shape[0], Wx.shape[2]).to(config.device)
        s = []

        # Bound values of the neuron parameters to plausible ranges
        alpha = torch.clamp(self.alpha, min=self.alpha_lim[0], max=self.alpha_lim[1])
        beta = torch.clamp(self.beta, min=self.beta_lim[0], max=self.beta_lim[1])
        a = torch.clamp(self.a, min=self.a_lim[0], max=self.a_lim[1])
        b = torch.clamp(self.b, min=self.b_lim[0], max=self.b_lim[1])
        if config.dropout > 0: self.V.weight.data = self.V.weight.data * mask.T
        V = self.V.weight.clone().fill_diagonal_(0) # Set diagonal elements of recurrent matrix to zero
        for t in range(Wx.shape[1]):
            wt = beta * wt + a * ut + b * st
            ut = alpha * (ut - st) + (1 - alpha) * (Wx[:, t, :] + torch.matmul(st, V) - wt)
            st = self.spike_fct(ut - config.threshold)
            s.append(st)
        return torch.stack(s, dim=1)

class ReadoutLayer(nn.Module):
    """
    This function implements a single layer of non-spiking Leaky Integrate and
    Fire (LIF) neurons, where the output consists of a cumulative sum of the
    membrane potential using a softmax function, instead of spikes.
    """
    def __init__(self, input_size, hidden_size, ):
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.alpha_lim = [np.exp(-1 / 5), np.exp(-1 / 25)]

        self.W = nn.Linear(self.input_size, self.hidden_size, bias=config.use_bias)
        self.alpha = nn.Parameter(torch.Tensor(self.hidden_size))
        nn.init.uniform_(self.alpha, self.alpha_lim[0], self.alpha_lim[1])

        self.normalize = False
        if config.normalization == "batchnorm":
            self.norm = nn.BatchNorm1d(self.hidden_size, momentum=0.05)
            self.normalize = True
        elif config.normalization == "layernorm":
            self.norm = nn.LayerNorm(self.hidden_size)
            self.normalize = True

        self.drop = nn.Dropout(p=config.pdrop)

    def forward(self, x, mask):
        Wx = self.W(x) # Feed-forward affine transformations (all steps in parallel)
        if self.normalize:
            _Wx = self.norm(Wx.reshape(Wx.shape[0] * Wx.shape[1], Wx.shape[2]))
            Wx = _Wx.reshape(Wx.shape[0], Wx.shape[1], Wx.shape[2])
        out = self.mem_update(Wx) # Wx.shape=[batch, nb_steps, output], out.shape=[batch, output]
        return out

    def mem_update(self, Wx):
        ut = torch.rand(Wx.shape[0], Wx.shape[2]).to(config.device)
        out = torch.zeros(Wx.shape[0], Wx.shape[2]).to(config.device)
        alpha = torch.clamp(self.alpha, min=self.alpha_lim[0], max=self.alpha_lim[1])

        for t in range(Wx.shape[1]):
            ut = alpha * ut + (1 - alpha) * Wx[:, t, :]
            out = out + F.softmax(ut, dim=1)
        return out

class SpikingDataset(Dataset):
    """
    Dataset class for the Spiking Heidelberg Digits (SHD) or Spiking Speech Commands (SSC) dataset.
    ---------
    split : str, Split of the SHD dataset, must be either "train" or "test".
    """

    def __init__(self, split,):
        self.device = "cpu"  # to allow pin memory
        self.max_time = 1.4
        self.time_bins = np.linspace(0, self.max_time, num=config.nb_steps)

        # Read data from h5py file
        filename = f"{config.data_folder}/{config.dataset_name}_{split}.h5"
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
        x_size = torch.Size([config.nb_steps, config.nb_inputs])

        x = torch.sparse.FloatTensor(x_idx, x_val, x_size).to(self.device)
        y = self.labels[index]

        return x.to_dense(), y

    def generateBatch(self, batch):
        xs, ys = zip(*batch)
        xs = torch.nn.utils.rnn.pad_sequence(xs, batch_first=True)
        xlens = torch.tensor([x.shape[0] for x in xs])
        ys = torch.LongTensor(ys).to(self.device)
        return xs, xlens, ys

def load_shd_or_ssc(split, shuffle=True, workers=0,):
    """
    This function creates a dataloader for a given split of the SHD or SSC datasets.
    ---------
    split : str, Split of dataset, must be either "train" or "test" for SHD. "train", "valid" or "test" for SSC.
    """
    if config.dataset_name not in ["shd", "ssc"]: raise ValueError(f"Invalid dataset name {config.dataset_name}")
    if split not in ["train", "valid", "test"]:   raise ValueError(f"Invalid split name {split}")
    if config.dataset_name == "shd" and split == "valid":
        logging.info("SHD does not have a validation split. Using test split.")
        split = "test"

    dataset = SpikingDataset(split)
    logging.info(f"Number of examples in {split} set: {len(dataset)}")

    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        collate_fn=dataset.generateBatch,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
    )
    return loader

def init_mask():
    a = torch.zeros((config.nb_hiddens, config.nb_hiddens), dtype=torch.int)
    for i in range(config.nb_cluster): 
        a[i*config.nb_per_cluster:(i+1)*config.nb_per_cluster, i*config.nb_per_cluster:(i+1)*config.nb_per_cluster] = 1.
    invalid_zeros = 1-(a==1).sum().item()/config.nb_hiddens**2
    if invalid_zeros < config.dropout:
        b = (torch.rand(config.nb_hiddens, config.nb_hiddens) > (config.dropout-invalid_zeros)/(1-invalid_zeros)).int() * (1-torch.eye(config.nb_hiddens, config.nb_hiddens, dtype=int))
        mask = a & b
        mask += torch.eye(config.nb_hiddens, config.nb_hiddens, dtype=int)
    else: mask = a
    return mask

class Experiment:
    """Training and testing models on SHD and SSC datasets."""
    def __init__(self,):
        # Initialize logging and output folders
        self.init_exp_folders()
        self.set_seed(config.seed)
        logging.FileHandler(filename=self.log_dir + "exp.log", mode="a", encoding=None, delay=False,)
        logging.basicConfig(filename=self.log_dir + "exp.log", level=logging.INFO, format="%(message)s", filemode='a',)
        logging.info("===== Exp configuration =====")
        for var in vars(config):
            if var[0] != '_': 
                logging.info(str(var) + ':\t\t' + str(vars(config)[var]))

        logging.info(f"\nDevice is set to {config.device}")
        logging.info(f"checkpoint_dir: {self.checkpoint_dir}\n")
        self.init_dataset()
        self.init_model()

        self.optimizer = torch.optim.Adam(self.net.parameters(), config.lr)
        self.scheduler = ReduceLROnPlateau(
            optimizer=self.optimizer,
            mode="max",
            factor=config.scheduler_factor,
            patience=config.scheduler_patience,
            min_lr=1e-6,
        )
        self.loss_fn = nn.CrossEntropyLoss()

    def set_seed(self, seed):
        if config.seed > 0:
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    def forward(self, trial):
        if not config.only_do_testing:
            self.init_model()
            train_accs, valid_accs = [], []
            # Initialize best accuracy
            if config.use_pretrained_model:
                logging.info("\n------ Using pretrained model ------\n")
                best_epoch, best_acc = self.valid_one_epoch(config.start_epoch, 0, 0)
            else:
                best_epoch, best_acc = 0, 0

            # Loop over epochs (training + validation)
            logging.info("\n------ Begin training ------\n")

            m1 = init_mask(); m2 = init_mask()
            # m1 = (torch.rand(config.hid, config.hid) > config.dropout).int() * (1-torch.eye(config.hid, config.hid)).int()
            # m2 = (torch.rand(config.hid, config.hid) > config.dropout).int() * (1-torch.eye(config.hid, config.hid)).int()
            mask = [m1.float().to(config.device), m2.float().to(config.device), 0]
            
            for e in range(best_epoch + 1, best_epoch + config.nb_epochs + 1):
                train_acc = self.train_one_epoch(e, mask); train_accs.append(train_acc)
                best_epoch, best_acc = self.valid_one_epoch(trial, e, mask, best_epoch, best_acc); valid_accs.append(best_acc)
                
                if config.dropout>0:
                    if (m1==0).sum().item()/config.nb_hiddens**2 <= config.dropout_stop or (m2==0).sum().item()/config.nb_hiddens**2 <= config.dropout_stop:
                        m1 = m1&((torch.rand(config.nb_hiddens, config.nb_hiddens) > config.dropout_stepping).int() * (1-torch.eye(config.nb_hiddens, config.nb_hiddens)).int())
                        m2 = m2&((torch.rand(config.nb_hiddens, config.nb_hiddens) > config.dropout_stepping).int() * (1-torch.eye(config.nb_hiddens, config.nb_hiddens)).int())
                        mask = [m1.float().to(config.device), m2.float().to(config.device), 0]

            logging.info(f"\nBest valid acc at epoch {best_epoch}: {best_acc}\n")
            logging.info("\n------ Training finished ------\n")

        # Loading best model
        # self.net = torch.load(f"{self.checkpoint_dir}/best_model-{best_acc}.tar", map_location=config.device)[0]
        # self.mask = torch.load(f"{self.checkpoint_dir}best_model-{best_acc}.tar", map_location=config.device)[1]
        # logging.info(f"Loading best model, epoch={best_epoch}, valid acc={best_acc}")

        # Test trained model
        # if config.dataset_name == "ssc":
        #     self.test_one_epoch(self.test_loader, self.mask)
        # else:
        #     self.test_one_epoch(self.valid_loader, self.mask)
        #     logging.info("\nThis dataset uses the same split for validation and testing.\n")
        
        return np.array(train_accs), np.array(valid_accs)

    def init_exp_folders(self):
        """define the output folders for the experiment."""
        # Check if path exists for loading pretrained model
        if config.use_pretrained_model:
            exp_folder = config.load_exp_folder
            self.load_path = exp_folder + "/checkpoints/best_model.pth"
            if not os.path.exists(self.load_path):
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), self.load_path)

        # Use given path for new model folder
        elif config.new_exp_folder is not None: exp_folder = config.new_exp_folder

        # Generate a path for new model from chosen config
        else:
            outname = config.dataset_name + "_" + config.neuron_type + "_"
            outname += str(config.nb_layers) + "lay" + str(config.nb_hiddens)
            outname += "_drop" + str(config.pdrop) + "_" + str(config.normalization)
            outname += "_bias" if config.use_bias else "_nobias"
            outname += "_bdir" if config.bidirectional else "_udir"
            outname += "_reg" if config.use_regularizers else "_noreg"
            outname += "_lr" + str(config.lr)
            exp_folder = "exp/test_exps/" + outname.replace(".", "_")

        # For a new model check that out path does not exist
        if not config.use_pretrained_model and os.path.exists(exp_folder):
            raise FileExistsError(errno.EEXIST, os.strerror(errno.EEXIST), exp_folder)

        # Create folders to store experiment
        self.log_dir = exp_folder + "/log/"
        self.checkpoint_dir = exp_folder + "/checkpoints/"
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        self.exp_folder = exp_folder

    def init_dataset(self):
        """This function prepares dataloaders for the desired dataset."""
        self.train_loader = load_shd_or_ssc(split="train", shuffle=True,)
        self.valid_loader = load_shd_or_ssc(split="valid", shuffle=False,)
        if config.dataset_name == "ssc":
            self.test_loader = load_shd_or_ssc(split="test", shuffle=False,)
        if config.use_augm:
            logging.warning("\nWarning: Data augmentation not implemented for SHD and SSC.\n")

    def init_model(self):
        if config.use_pretrained_model:
            self.net = torch.load(self.load_path, map_location=config.device)
            logging.info(f"\nLoaded model at: {self.load_path}\n {self.net}\n")

        elif config.neuron_type in ["LIF", "adLIF", "RLIF", "RadLIF"]:
            self.net = SNN().to(config.device)
            logging.info(f"\nCreated new spiking model:\n {self.net}\n")

        self.nb_params = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
        logging.info(f"Total number of trainable parameters is {self.nb_params}")

    def train_one_epoch(self, e, mask):
        start = time.time()
        self.net.train()
        losses, accs, epoch_spike_rate = [], [], 0

        for step, (x, _, y) in enumerate(self.train_loader):
            x = x.to(config.device); y = y.to(config.device)
            output, firing_rates, all_spikes = self.net(x, mask)
            loss_val = self.loss_fn(output, y)

            # Spike activity
            epoch_spike_rate += torch.mean(firing_rates)
            if config.use_regularizers:
                reg_quiet = F.relu(config.reg_fmin - firing_rates).sum()
                reg_burst = F.relu(firing_rates - config.reg_fmax).sum()
                loss_val += config.reg_factor * (reg_quiet + reg_burst)
            
            # clustering for first layer
            cluster_ins1, cluster_outs1 = [], []
            global_mean = 0
            for i in range(config.nb_cluster):
                cluster_mean = self.net.snn[0].V.weight[i*config.nb_per_cluster:(i+1)*config.nb_per_cluster, i*config.nb_per_cluster:(i+1)*config.nb_per_cluster].mean(1)
                global_mean += cluster_mean
                cluster_in = F.cosine_similarity(self.net.snn[0].V.weight[i*config.nb_per_cluster:(i+1)*config.nb_per_cluster, i*config.nb_per_cluster:(i+1)*config.nb_per_cluster], cluster_mean, dim=1)
                cluster_ins1.append(cluster_in.mean())
            for i in range(config.nb_cluster):
                cluster_outs1.append(F.cosine_similarity(self.net.snn[0].V.weight[i*config.nb_per_cluster:(i+1)*config.nb_per_cluster, i*config.nb_per_cluster:(i+1)*config.nb_per_cluster].mean(1), global_mean, dim=0))
            cin1 = torch.var(torch.stack(cluster_ins1)); cout1 = torch.var(torch.stack(cluster_outs1))
            
            # for second layer
            cluster_ins2, cluster_outs2 = [], []
            global_mean = 0
            for i in range(config.nb_cluster):
                cluster_mean = self.net.snn[1].V.weight[i*config.nb_per_cluster:(i+1)*config.nb_per_cluster, i*config.nb_per_cluster:(i+1)*config.nb_per_cluster].mean(1)
                global_mean += cluster_mean
                cluster_in = F.cosine_similarity(self.net.snn[1].V.weight[i*config.nb_per_cluster:(i+1)*config.nb_per_cluster, i*config.nb_per_cluster:(i+1)*config.nb_per_cluster], cluster_mean, dim=1)
                cluster_ins2.append(cluster_in.mean())
            for i in range(config.nb_cluster):
                cluster_outs2.append(F.cosine_similarity(self.net.snn[1].V.weight[i*config.nb_per_cluster:(i+1)*config.nb_per_cluster, i*config.nb_per_cluster:(i+1)*config.nb_per_cluster].mean(1), global_mean, dim=0))
            cin2 = torch.var(torch.stack(cluster_ins2)); cout2 = torch.var(torch.stack(cluster_outs2))
            if config.clustering:
                print('1',loss_val)
                loss_val += config.clustering_factor[0] * (F.relu(config.cin_minmax[0] - cin1) + F.relu(cin1 - config.cin_minmax[1]))
                print('2',loss_val)
                loss_val += config.clustering_factor[0] * (F.relu(config.cin_minmax[0] - cin2) + F.relu(cin2 - config.cin_minmax[1]))
                print('3',loss_val)
                loss_val += config.clustering_factor[1] * (F.relu(config.cout_minmax[0] - cout1) + F.relu(cout1 - config.cout_minmax[1]))
                loss_val += config.clustering_factor[1] * (F.relu(config.cout_minmax[0] - cout2) + F.relu(cout2 - config.cout_minmax[1]))
                print("%.6f, %.6f, %.6f, %.6f, %.6f, %.6f"%(cin1.item(), cin2.item(), cout1.item(), cout2.item(), cout1.item()/cin1.item(), cout2.item()/cin2.item()))

            losses.append(loss_val.item())
            self.optimizer.zero_grad()
            loss_val.backward()
            self.optimizer.step()

            pred = torch.argmax(output, dim=1)
            acc = np.mean((y == pred).detach().cpu().numpy())
            accs.append(acc)

        # Learning rate of whole epoch
        current_lr = self.optimizer.param_groups[-1]["lr"]
        train_loss = np.mean(losses)
        train_acc = np.mean(accs)
        epoch_spike_rate /= step
        elapsed = str(timedelta(seconds=time.time() - start))[5:]
        logging.info(f"Epoch {e}: train loss={train_loss:.4f}, acc={train_acc:.4f}, fr={epoch_spike_rate:.4f}, lr={current_lr:.4f}, time={elapsed}, cin={cin1.item():.6f}, {cin2.item():.6f}, cout={cout1.item():.6f}, {cout2.item():.6f}")
        return train_acc
    
    def valid_one_epoch(self, trial, e, mask, best_epoch, best_acc):
        start = time.time()
        with torch.no_grad():
            self.net.eval()
            losses, accs, epoch_spike_rate = [], [], 0

            for step, (x, _, y) in enumerate(self.valid_loader):
                x += torch.rand_like(x) * config.noise_test
                x = x.to(config.device); y = y.to(config.device)
                output, firing_rates, all_spikes = self.net(x, mask)
                loss_val = self.loss_fn(output, y)
                losses.append(loss_val.item())

                pred = torch.argmax(output, dim=1)
                accs.append(np.mean((y == pred).detach().cpu().numpy()))
                epoch_spike_rate += torch.mean(firing_rates)

            valid_loss = np.mean(losses)
            valid_acc = np.mean(accs)
            epoch_spike_rate /= step
            elapsed = str(timedelta(seconds=time.time() - start))[5:]
            sparsity = ((mask[0]==0).sum().item()/config.nb_hiddens**2 + (mask[1]==0).sum().item()/config.nb_hiddens**2)/2
            logging.info(f"Epoch {e}: valid loss={valid_loss:.4f}, acc={valid_acc:.4f}, fr={epoch_spike_rate:.4f}, mask={sparsity:.4f}, time={elapsed}")

            self.scheduler.step(valid_acc)

            if e % config.ckpt_freq == 0:
                torch.save([self.net, mask], self.checkpoint_dir+'/model-{:d}-{:d}-{:.4f}.tar'.format(trial, e, valid_acc))
            # Update best epoch and accuracy
            if valid_acc > best_acc:
                best_acc = valid_acc; best_epoch = e
                if valid_acc > 0.85: torch.save([self.net, mask], f"{self.checkpoint_dir}/best_model-{valid_acc}.tar")
                logging.info(f"\nBest model saved with valid acc={valid_acc}")

            logging.info("\n-----------------------------\n")
            return best_epoch, best_acc

    def test_one_epoch(self, test_loader, mask):
        with torch.no_grad():
            self.net.eval()
            losses, accs, epoch_spike_rate = [], [], 0

            logging.info("\n------ Begin Testing ------\n")
            for step, (x, _, y) in enumerate(test_loader):
                x = x.to(config.device); y = y.to(config.device)
                output, firing_rates, all_spikes = self.net(x, mask)
                loss_val = self.loss_fn(output, y)
                losses.append(loss_val.item())
                pred = torch.argmax(output, dim=1)
                acc = np.mean((y == pred).detach().cpu().numpy())
                accs.append(acc)
                epoch_spike_rate += torch.mean(firing_rates)

            test_loss = np.mean(losses)
            test_acc = np.mean(accs)
            epoch_spike_rate /= step
            logging.info(f"Test loss={test_loss}, acc={test_acc}, mean act rate={epoch_spike_rate}")
            logging.info("\n-----------------------------\n")

def plot_errorbar(train_acc_log, test_acc_log, file_name):
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

    plt.plot(list(range(config.nb_epochs)), train_mean, color='deeppink', label='train')
    plt.fill_between(list(range(config.nb_epochs)), train_mean-train_std, train_mean+train_std, color='deeppink', alpha=0.2)
    # plt.fill_between(list(range(config.epoch)), train_min, train_max, color='violet', alpha=0.2)

    plt.plot(list(range(config.nb_epochs)), test_mean, color='blue', label='test')
    plt.fill_between(list(range(config.nb_epochs)), test_mean-test_std, test_mean+test_std, color='blue', alpha=0.2)
    # plt.fill_between(list(range(config.epoch)), test_min, test_max, color='blue', alpha=0.2)

    plt.legend()
    plt.grid()
    # plt.axis([-5, 105, 75, 95])
    plt.savefig(file_name)
    # plt.show()

if __name__ == "__main__":
    experiment = Experiment()
    log = np.zeros((2, config.nb_epochs, config.trial))
    for i in range(config.trial):
        logging.info(f"\n---------------Trial:{i+1}---------------\n")
        experiment.set_seed(config.seed + i + 1)
        train_accs, valid_accs = experiment.forward(i+1)
        log[0,:,i] = train_accs
        log[1,:,i] = valid_accs
    plot_errorbar(log[0], log[1], './fig/'+ config.date +'.pdf')