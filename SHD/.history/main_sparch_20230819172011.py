'''
首次修改2023年8月18日20:53:29

'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import time, warnings, os, h5py, logging
warnings.filterwarnings("ignore")
from datetime import timedelta
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
logger = logging.getLogger(__name__)

class config:
    date = time.strftime("%Y-%m-%d-%H-%M-%S/", time.localtime(time.time()))[5:16]
    save_dir = './log/' + date
    dataset_name = 'shd'
    data_folder = './data/raw/'
    input_dim = 700
    output_dim = 20
    nb_steps = 100
    trials = 5
    scheduler_patience = 1
    scheduler_factor = 0.7
    
    batch_size = 512
    nb_epochs = 100
    lr = 1e-2
    weight_decay = 1e-5
    reg_factor = 0.5
    reg_fmin = 0.01
    reg_fmax = 0.1
    
    seed = round(time.time())
    ckpt_freq = 10
    threshold = 1.0
    
    smoothing = 0.1
    pdrop = 0.1
    normalization = 'batchnorm'
    train_input = True
    nb_hiddens = 1024
    noise_test = 0.0
    
    device = 'cuda'

##########################################################
########### define surrogate gradient function ###########
def gaussian(x, mu=0., sigma=.5):
    return torch.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / torch.sqrt(2 * torch.tensor(torch.pi)) / sigma

class ActFun_adp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):  # input = membrane potential- threshold
        ctx.save_for_backward(input)
        return input.gt(0).float()

    @staticmethod
    def backward(ctx, grad_output):  # approximate the gradients
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        if config.gradient_type == 'G':
            temp = torch.exp(-(input**2)/(2*config.lens**2))/torch.sqrt(2*torch.tensor(torch.pi))/config.lens
        elif config.gradient_type == 'MG':
            temp = gaussian(input, mu=0., sigma=config.lens) * (1. + config.hight) \
                - gaussian(input, mu=config.lens, sigma=config.scale * config.lens) * config.hight \
                - gaussian(input, mu=-config.lens, sigma=config.scale * config.lens) * config.hight
        elif config.gradient_type =='linear':
            temp = F.relu(1-input.abs())
        elif config.gradient_type == 'slayer':
            temp = torch.exp(-5*input.abs())
        return grad_input * temp.float() * config.gamma
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

class RC(nn.Module):
    def __init__(self):
        super(RC, self).__init__()
        # Fixed params
        self.alpha_lim = [np.exp(-1 / 5), np.exp(-1 / 25)]
        self.beta_lim = [np.exp(-1 / 30), np.exp(-1 / 120)]
        self.a_lim = [-1.0, 1.0]
        self.b_lim = [0.0, 2.0]
        self.spike_fct = SpikeFunctionBoxcar.apply
        
        # Trainable parameters
        self.W = nn.Linear(config.input_dim, config.nb_hiddens, bias=True)
        self.V = nn.Linear(config.nb_hiddens, config.nb_hiddens, bias=False)
        self.alpha = nn.Parameter(torch.Tensor(config.nb_hiddens))
        self.beta = nn.Parameter(torch.Tensor(config.nb_hiddens))
        self.a = nn.Parameter(torch.Tensor(config.nb_hiddens))
        self.b = nn.Parameter(torch.Tensor(config.nb_hiddens))
        
        self.read = nn.Linear(config.nb_hiddens, config.output_dim, bias=True)
        self.alpha_read = nn.Parameter(torch.Tensor(config.output_dim))

        nn.init.uniform_(self.alpha, self.alpha_lim[0], self.alpha_lim[1])
        nn.init.uniform_(self.beta, self.beta_lim[0], self.beta_lim[1])
        nn.init.uniform_(self.a, self.a_lim[0], self.a_lim[1])
        nn.init.uniform_(self.b, self.b_lim[0], self.b_lim[1])
        nn.init.orthogonal_(self.V.weight)
        nn.init.uniform_(self.alpha_read, self.alpha_lim[0], self.alpha_lim[1])

        # Initialize normalization
        self.normalize = False
        if config.normalization == "batchnorm":
            self.norm = nn.BatchNorm1d(config.nb_hiddens, momentum=0.05)
            self.norm_read = nn.BatchNorm1d(config.output_dim, momentum=0.05)
            self.normalize = True
        elif config.normalization == "layernorm":
            self.norm = nn.LayerNorm(config.nb_hiddens)
            self.norm_read = nn.LayerNorm(config.output_dim)
            self.normalize = True
        self.drop = nn.Dropout(p=config.pdrop)
        
        if not config.train_input:
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
            st = self.spike_fct(ut - config.threshold)
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
        all_spikes.append(s)
        s = self.drop(s)
        
        
        Wx_ = self.read(s)
        if self.normalize:
            _Wx_ = self.norm_read(Wx_.reshape(Wx_.shape[0] * Wx_.shape[1], Wx_.shape[2]))
            Wx_ = _Wx_.reshape(Wx_.shape[0], Wx_.shape[1], Wx_.shape[2])
        out = self.readout_cell(Wx_, self.alpha_read)

        firing_rates = torch.cat(all_spikes, dim=2).mean(dim=(0, 1)) # Compute mean firing rate of each spiking neuron
        return out, firing_rates, torch.cat(all_spikes, dim=2) # (batch, timestep, hidden_size)
    
#####################################
########### load SHD data ###########
def load_shd_or_ssc():
    train_X = np.load('data/trainX_10ms.npy')
    train_y = np.load('data/trainY_10ms.npy').astype(float)

    test_X = np.load('data/testX_10ms.npy')
    test_y = np.load('data/testY_10ms.npy').astype(float)

    print('dataset shape: ', train_X.shape)
    print('dataset shape: ', test_X.shape)

    tensor_trainX = torch.Tensor(train_X)  # transform to torch tensor
    tensor_trainY = torch.Tensor(train_y)
    train_dataset = torch.utils.data.TensorDataset(tensor_trainX, tensor_trainY)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    tensor_testX = torch.Tensor(test_X)  # transform to torch tensor
    tensor_testY = torch.Tensor(test_y)
    test_dataset = torch.utils.data.TensorDataset(tensor_testX, tensor_testY)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    return train_loader, test_loader

################################################
########### define training pipeline ###########

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def train(trial, model, optimizer, criterion, num_epochs, train_loader, test_loader, scheduler):
    logging.info(f'**************  Trial {trial}  **************')
    model.train()
    train_accs, test_accs = [], []

    mask = 0 # [m1.float().to(config.device), m2.float().to(config.device)]
    
    torch.save(model, config.save_dir+'/before-train-{:d}-{:d}-{:.2f}.tar'.format(trial, 0, 0))
    best_acc, best_epoch = 85, 0
    for e in range(1, num_epochs+1):
        now = time.time()
        losses, correct, total, epoch_spike_rate = [], 0, 0, 0
        for i, (samples, labels) in enumerate(tqdm(train_loader)):
            # samples = samples.requires_grad_().to(device)
            samples = torch.sign(samples.clamp(min=0)).to(config.device) # all pixels should be 0 or 1
            labels = labels.long().to(config.device)
            
            outputs, firing_rates, all_spikes = model(samples, mask)
            
            # label smoothing
            if config.smoothing>0:
                with torch.no_grad():
                    true_dist = torch.zeros_like(outputs)
                    true_dist.fill_(config.smoothing / (config.output_dim - 1))
                    true_dist.scatter_(1, labels.unsqueeze(1), 1.0 - config.smoothing)
                loss = criterion(outputs, true_dist)
            else:
                loss = criterion(outputs, labels)
            
            epoch_spike_rate += torch.mean(firing_rates)
            reg_quiet = F.relu(config.reg_fmin - firing_rates).sum()
            reg_burst = F.relu(firing_rates - config.reg_fmax).sum()
            loss += config.reg_factor * (reg_quiet + reg_burst)
            
            # print(all_spikes.max(), all_spikes.min())
            print( -(all_spikes.mean(0) * (all_spikes.mean(0)+1e-4).log()).sum() )
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted.cpu() == labels.long().cpu()).sum()
        
        current_lr = optimizer.param_groups[-1]["lr"]
        tr_loss = np.mean(losses)
        tr_acc = 100. * correct.numpy() / total
        ts_acc, ts_loss = test(model, test_loader, mask, criterion)
        train_accs.append(tr_acc)
        test_accs.append(ts_acc)
        scheduler.step(ts_acc)
        elapsed = str(timedelta(seconds=time.time() - now))[5:]
        
        logging.info(f"Epoch {e}: loss={tr_loss:.4f}|{ts_loss:.4f}, acc={tr_acc:.4f}|{ts_acc:.4f}, fr={epoch_spike_rate:.4f}, lr={current_lr:.4f}, time={elapsed}")
        if ts_acc > best_acc:
            best_acc = ts_acc; best_epoch = e
            torch.save(model, f"{config.save_dir}/best_model_{trial}_{ts_acc}.pth")
            logging.info(f"\nBest model saved with valid acc={ts_acc}")
        if e % config.ckpt_freq==0:
            torch.save(model, config.save_dir+'/model-{:d}-{:d}-{:.2f}.tar'.format(trial, e, ts_acc))
        logging.info("\n-----------------------------\n")
        
    
    logging.info(f"\nBest valid acc at epoch {best_epoch}: {best_acc}")
    logging.info("\n------ Training finished ------\n")
    return np.array(train_accs), np.array(test_accs)

def test(model, dataloader, mask, criterion):
    model.eval()
    with torch.no_grad():
        losses, correct, total = [], 0, 0
        for images, labels in dataloader:
            images = torch.sign(images.clamp(min=0)) # all pixels should be 0 or 1
            if config.noise_test>0:
                images += torch.rand_like(images) * config.noise_test
            outputs, firing_rates, all_spikes = model(images.to(config.device), mask)
            loss_val = criterion(outputs, labels.long().to(config.device))
            losses.append(loss_val.item())
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted.cpu() == labels.long().cpu()).sum()
        accuracy = 100. * correct.numpy() / total
        loss = np.mean(losses)
    return accuracy, loss


def multiple_trial(train_loader, test_loader):
    os.makedirs(config.save_dir) if not os.path.exists(config.save_dir) else None
    logging.FileHandler(filename=config.save_dir + "/exp.log", mode="a", encoding=None, delay=False,)
    logging.basicConfig(filename=config.save_dir + "/exp.log", level=logging.INFO, format="%(message)s",)
    
    logging.info("===== Exp configuration =====")
    for var in vars(config):
        if var[0] != '_': 
            logging.info(str(var) + ':\t\t' + str(vars(config)[var]))
            
    train_acc_log = np.zeros((config.nb_epochs, config.trials))
    test_acc_log = np.zeros((config.nb_epochs, config.trials))
    for i in range(config.trials):
        set_seed(config.seed+i)
        model = RC().to(config.device)
        if i == 0:
            nb_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logging.info(f"\nCreated new spiking model:\n {model}\n")
            logging.info(f"Total number of trainable parameters is {nb_params}")
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config.lr, weight_decay=config.weight_decay)
        # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.lr)
        scheduler = ReduceLROnPlateau(
            optimizer=optimizer,
            mode="max",
            factor=config.scheduler_factor,
            patience=config.scheduler_patience,
            min_lr=1e-6,
        )
        train_acc, test_acc = train(i, model, optimizer, criterion, config.nb_epochs, train_loader, test_loader, scheduler)
        
        train_acc_log[:,i] = train_acc
        test_acc_log[:,i] = test_acc
    return train_acc_log, test_acc_log

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
    # plt.fill_between(list(range(config.nb_epochs)), train_min, train_max, color='violet', alpha=0.2)

    plt.plot(list(range(config.nb_epochs)), test_mean, color='blue', label='test')
    plt.fill_between(list(range(config.nb_epochs)), test_mean-test_std, test_mean+test_std, color='blue', alpha=0.2)
    # plt.fill_between(list(range(config.nb_epochs)), test_min, test_max, color='blue', alpha=0.2)

    plt.legend()
    plt.grid()
    # plt.axis([-5, 105, 75, 95])
    plt.savefig(file_name)
    # plt.show()


###################################
########### start train ###########
if __name__ == "__main__":
    print('Start')
    #####################################
    ########### load SHD data ###########
    train_loader, test_loader = load_shd_or_ssc()
    # nmnist_train = NMNIST('./data/', train=True, data_type='frame', frames_number=10, split_by='number')
    # nmnist_test = NMNIST('./data/', train=False, data_type='frame', frames_number=10, split_by='number')
    # train_loader = torch.utils.data.DataLoader(dataset=nmnist_train, batch_size=config.batch_size, shuffle=True, drop_last=False, num_workers=0)
    # test_loader = torch.utils.data.DataLoader(dataset=nmnist_test, batch_size=config.batch_size, shuffle=False, drop_last=False, num_workers=0)
    print('Finish loading data')
    train_acc_log, test_acc_log = multiple_trial(train_loader, test_loader)
    plot_errorbar(train_acc_log, test_acc_log, config.save_dir+ 'fig' +'.pdf')