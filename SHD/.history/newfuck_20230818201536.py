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

class Config:
    date = time.strftime("%Y-%m-%d-%H-%M-%S/", time.localtime(time.time()))[5:16]
    new_exp_folder = './log/' + date
    dataset_name = 'shd'
    data_folder = './data/raw/'
    input_dim = 700
    output_dim = 20
    
    batch_size = 512
    nb_epochs = 30
    lr = 1e-2
    scheduler_patience = 1
    scheduler_factor = 0.7
    reg_factor = 0.5
    reg_fmin = 0.01
    reg_fmax = 0.2
    nb_steps = 50
    trial = 5
    seed = round(time.time())
    ckpt_freq = 5
    threshold = 1.0
    
    pdrop = 0.1
    normalization = 'batchnorm'
    train_input = True
    nb_hiddens = 1024
    noise_test = 0.0
    
    device = 'cuda'
    
    
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
    The function returns the outputs of the last spiking or readout layer with shape (batch, time, feats) or (batch, feats) respectively, 
    as well as the firing rates of all hidden neurons with shape (num_layers*feats).
    """
    def __init__(self, ):
        super().__init__()
        # Fixed params
        self.alpha_lim = [np.exp(-1 / 5), np.exp(-1 / 25)]
        self.beta_lim = [np.exp(-1 / 30), np.exp(-1 / 120)]
        self.a_lim = [-1.0, 1.0]
        self.b_lim = [0.0, 2.0]
        self.spike_fct = SpikeFunctionBoxcar.apply
        
        # Trainable parameters
        self.W = nn.Linear(Config.input_dim, Config.nb_hiddens, bias=True)
        self.V = nn.Linear(Config.nb_hiddens, Config.nb_hiddens, bias=False)
        self.alpha = nn.Parameter(torch.Tensor(Config.nb_hiddens))
        self.beta = nn.Parameter(torch.Tensor(Config.nb_hiddens))
        self.a = nn.Parameter(torch.Tensor(Config.nb_hiddens))
        self.b = nn.Parameter(torch.Tensor(Config.nb_hiddens))
        
        self.W_read = nn.Linear(Config.nb_hiddens, Config.output_dim, bias=True)
        self.alpha_read = nn.Parameter(torch.Tensor(Config.output_dim))

        nn.init.uniform_(self.alpha, self.alpha_lim[0], self.alpha_lim[1])
        nn.init.uniform_(self.beta, self.beta_lim[0], self.beta_lim[1])
        nn.init.uniform_(self.a, self.a_lim[0], self.a_lim[1])
        nn.init.uniform_(self.b, self.b_lim[0], self.b_lim[1])
        nn.init.orthogonal_(self.V.weight)
        nn.init.uniform_(self.alpha_read, self.alpha_lim[0], self.alpha_lim[1])

        # Initialize normalization
        self.normalize = False
        if normalization == "batchnorm":
            self.norm = nn.BatchNorm1d(Config.nb_hiddens, momentum=0.05)
            self.norm_read = nn.BatchNorm1d(Config.output_dim, momentum=0.05)
            self.normalize = True
        elif normalization == "layernorm":
            self.norm = nn.LayerNorm(Config.nb_hiddens)
            self.norm_read = nn.LayerNorm(Config.output_dim)
            self.normalize = True
        self.drop = nn.Dropout(p=pdrop)
        
        if not self.train_input:
            for name, p in self.named_parameters():
                if 'W' in name: p.requires_grad = False
    
    def radlif_cell(self, Wx, mask, alpha, beta, a, b, V):
        ut = torch.rand(Wx.shape[0], Wx.shape[2]).to(Wx.device)
        wt = torch.rand(Wx.shape[0], Wx.shape[2]).to(Wx.device)
        st = torch.rand(Wx.shape[0], Wx.shape[2]).to(Wx.device)
        s = []
        # Bound values of parameters to plausible ranges
        alpha_ = torch.clamp(alpha, min=self.alpha_lim[0], max=self.alpha_lim[1])
        beta_ = torch.clamp(beta, min=self.beta_lim[0], max=self.beta_lim[1])
        a_ = torch.clamp(a, min=self.a_lim[0], max=self.a_lim[1])
        b_ = torch.clamp(b, min=self.b_lim[0], max=self.b_lim[1])
        # if self.dropout > 0: self.V.weight.data = self.V.weight.data * mask.T
        V_ = V.weight.clone().fill_diagonal_(0)
        for t in range(Wx.shape[1]):
            wt = beta_ * wt + a_ * ut + b_ * st
            ut = alpha_ * (ut - st) + (1 - alpha_) * (Wx[:, t, :] + torch.matmul(st, V_) - wt)
            st = self.spike_fct(ut - Config.threshold)
            s.append(st)
        return torch.stack(s, dim=1)
    
    def readout_cell(self, Wx, alpha):
        ut = torch.rand(Wx.shape[0], Wx.shape[2]).to(Wx.device)
        out = torch.zeros(Wx.shape[0], Wx.shape[2]).to(Wx.device)
        alpha_ = torch.clamp(alpha, min=self.alpha_lim[0], max=self.alpha_lim[1]) # Bound values of the neuron parameters to plausible ranges
        for t in range(Wx.shape[1]):
            ut = alpha * ut + (1 - alpha_) * Wx[:, t, :]
            out = out + F.softmax(ut, dim=1)
        return out
    
    def forward(self, x, mask):
        all_spikes = []
        
        Wx = self.W(x) # (all steps in parallel)
        if self.normalize:
            _Wx = self.norm(Wx.reshape(Wx.shape[0] * Wx.shape[1], Wx.shape[2]))
            Wx = _Wx.reshape(Wx.shape[0], Wx.shape[1], Wx.shape[2])
        s = self.radlif_cell(Wx, mask, self.alpha, self.beta, self.a, self.b, self.V)
        s = self.drop(s)
        all_spikes.append(s)
        
        Wx_ = self.W_read(s)
        if self.normalize:
            _Wx_ = self.norm(Wx_.reshape(Wx_.shape[0] * Wx_.shape[1], Wx_.shape[2]))
            Wx_ = _Wx_.reshape(Wx_.shape[0], Wx_.shape[1], Wx_.shape[2])
        out = self.readout_cell(Wx_, self.alpha_read)

        firing_rates = torch.cat(all_spikes, dim=2).mean(dim=(0, 1)) # Compute mean firing rate of each spiking neuron
        return out, firing_rates, all_spikes
        
class SpikingDataset(Dataset):
    def __init__(self, split,):
        # Fixed parameters
        self.device = "cpu"  # to allow pin memory
        self.max_time = 1.4
        self.time_bins = np.linspace(0, self.max_time, num=Config.nb_steps)

        filename = f"{Config.data_folder}/{Config.dataset_name}_{split}.h5"
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
        x_size = torch.Size([Config.nb_steps, Config.input_dim])

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
    if Config.dataset_name == "shd" and split == "valid":
        logging.info("SHD does not have a validation split. Using test split.")
        split = "test"

    dataset = SpikingDataset(split)
    logging.info(f"Number of examples in {split} set: {len(dataset)}")

    loader = DataLoader(
        dataset,
        batch_size=Config.batch_size,
        collate_fn=dataset.generateBatch,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
    )
    return loader

class Experiment:
    def __init__(self, ):
        self.nb_cluster = args.nb_cluster
        self.nb_per_cluster = int(args.nb_hiddens / args.nb_cluster)

        self.set_seed(Config.seed)
        self.init_exp_folders()
        
        # load data
        self.train_loader = load_shd_or_ssc(split="train", shuffle=True,)
        self.valid_loader = load_shd_or_ssc(split="valid", shuffle=False,)
        self.test_loader = load_shd_or_ssc(split="test", shuffle=False,)
        # init log
        logging.FileHandler(filename=self.log_dir + "exp.log", mode="a", encoding=None, delay=False,)
        logging.basicConfig(filename=self.log_dir + "exp.log", level=logging.INFO, format="%(message)s",)
        
        self.net = SNN().to(Config.device)
        self.nb_params = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
        logging.info(f"\nCreated new spiking model:\n {self.net}\n")
        logging.info(f"Total number of trainable parameters is {self.nb_params}")
        logging.info(f"\nDevice is set to {Config.device}\n")

        self.optimizer = torch.optim.Adam(self.net.parameters(), Config.lr)
        self.scheduler = ReduceLROnPlateau(
            optimizer=self.optimizer,
            mode="max",
            factor=Config.scheduler_factor,
            patience=Config.scheduler_patience,
            min_lr=1e-6,
        )
        self.loss_fn = nn.CrossEntropyLoss()

    def set_seed(self, seed):
        if seed > 0:
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # torch.backends.cudnn.deterministic = True
            # torch.backends.cudnn.benchmark = False

    def forward(self, trial):
        self.net = SNN().to(Config.device)
        train_accs, valid_accs = [], []
        best_epoch, best_acc = 0, 0

        logging.info("\n------ Begin training ------\n")
        # m1 = init_mask(self.nb_hiddens, self.nb_cluster); m2 = init_mask(self.nb_hiddens, self.nb_cluster)
        # m1 = (torch.rand(Config.hid, Config.hid) > Config.dropout).int() * (1-torch.eye(Config.hid, Config.hid)).int()
        # m2 = (torch.rand(Config.hid, Config.hid) > Config.dropout).int() * (1-torch.eye(Config.hid, Config.hid)).int()
        mask = 0 # [m1.float().to(Config.device), m2.float().to(Config.device), 0]
        for e in range(1, Config.nb_epochs + 1):
            train_acc = self.train_one_epoch(e, mask); train_accs.append(train_acc)
            best_epoch, best_acc = self.valid_one_epoch(trial, e, mask, best_epoch, best_acc); valid_accs.append(best_acc)

        logging.info(f"\nBest valid acc at epoch {best_epoch}: {best_acc}\n")
        logging.info("\n------ Training finished ------\n")
        
        self.net = torch.load(f"{self.checkpoint_dir}/best_model_{trial}_{best_acc}.pth", map_location=Config.device)
        logging.info(f"Loading best model, epoch={best_epoch}, valid acc={best_acc}")

        # Test trained model
        if Config.dataset_name == "ssc": self.test_one_epoch(self.test_loader)
        else:
            self.test_one_epoch(self.valid_loader)
            logging.info("\nThis dataset uses the same split for validation and testing.\n")
        return np.array(train_accs), np.array(valid_accs)

    def init_exp_folders(self):
        if self.new_exp_folder is not None: 
            exp_folder = Config.new_exp_folder
        if os.path.exists(exp_folder):
            raise FileExistsError(errno.EEXIST, os.strerror(errno.EEXIST), exp_folder)

        self.log_dir = exp_folder + "/log/"
        self.checkpoint_dir = exp_folder + "/checkpoints/"
        if not os.path.exists(self.log_dir):         os.makedirs(self.log_dir)
        if not os.path.exists(self.checkpoint_dir):  os.makedirs(self.checkpoint_dir)
        self.exp_folder = exp_folder

    def train_one_epoch(self, e, mask):
        start = time.time()
        self.net.train()
        losses, accs, epoch_spike_rate = [], [], 0

        for step, (x, _, y) in enumerate(self.train_loader):
            x = x.to(Config.device); y = y.to(Config.device)
            output, firing_rates, all_spikes = self.net(x, mask)
            loss_val = self.loss_fn(output, y)

            epoch_spike_rate += torch.mean(firing_rates)
            reg_quiet = F.relu(Config.reg_fmin - firing_rates).sum()
            reg_burst = F.relu(firing_rates - Config.reg_fmax).sum()
            loss_val += Config.reg_factor * (reg_quiet + reg_burst)

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
                x += torch.rand_like(x) * Config.noise_test
                x = x.to(Config.device); y = y.to(Config.device)
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
                x += torch.rand_like(x) * Config.noise_test
                x = x.to(Config.device); y = y.to(Config.device)
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

    plt.plot(list(range(Config.nb_epochs)), train_mean, color='deeppink', label='train')
    plt.fill_between(list(range(cofig.nb_epochs)), train_mean-train_std, train_mean+train_std, color='deeppink', alpha=0.2)
    # plt.fill_between(list(range(args.epoch)), train_min, train_max, color='violet', alpha=0.2)

    plt.plot(list(range(Config.nb_epochs)), test_mean, color='blue', label='test')
    plt.fill_between(list(range(Config.nb_epochs)), test_mean-test_std, test_mean+test_std, color='blue', alpha=0.2)
    # plt.fill_between(list(range(args.epoch)), test_min, test_max, color='blue', alpha=0.2)

    plt.legend()
    plt.grid()
    # plt.axis([-5, 105, 75, 95])
    plt.savefig(file_name)
    # plt.show()

if __name__ == "__main__":
    experiment = Experiment()

    log = np.zeros((2, Config.nb_epochs, Config.trial))
    for i in range(Config.trial):
        logging.info(f"\n---------------Trial:{i+1}---------------\n")
        experiment.set_seed(Config.seed + i + 1)
        train_accs, valid_accs = experiment.forward(i+1)
        log[0,:,i] = train_accs
        log[1,:,i] = valid_accs
    plot_errorbar(Config, log[0], log[1], './fig/'+ Config.date +'.pdf')