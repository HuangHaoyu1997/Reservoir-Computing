'''
首次修改2023年7月12日23:59:14

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
    seed = 20
    input = 700
    hid = 128         # number of RC Neurons
    output = 20
    nb_steps = 250  # Number of steps to unroll
    
    scheduler_patience = 2      # Number of epochs without progress before the learning rate gets decreased.
    scheduler_factor = 0.7      # Factor between 0 and 1 by which the learning rate gets decreased when the scheduler patience is reached.
    
    thr = 0.5
    b_j0 = 0.01       # thr baseline
    dt = 1
    R_m = 1
    decay = 0.5
    rst = 0.05
    lens = 0.5
    gamma = 0.5       # gradient scale 
    gradient_type = 'MG' # 'G', 'slayer', 'linear' 窗型函数
    scale = 6.        # special for 'MG'
    hight = 0.15      # special for 'MG'
    input_learn = True # learnable input layer
    epoch = 50
    trials = 5        # try on 5 different seeds
    batch = 256
    lr = 1e-2
    l1 = 0.0000 # 0.0003
    l1_targ = 2000
    fr_norm = 0.005
    fr_min = 0.01
    fr_max = 0.10
    dropout = 0.75
    dropout_stepping = 0.04
    dropout_stop = 0.95
    weight_decay = 1e-4
    smoothing = 0.1
    noise_test = 0.0
    norm = False      # add layer norm before each layer
    shortcut = False
    small_init = False
    ckpt_freq = 10    # every 10 epoch save model
    device = torch.device('cuda')

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
act_fun_adp = ActFun_adp.apply

#######################################
########### define RC model ###########

class RC(nn.Module):
    def __init__(self):
        super(RC, self).__init__()
        self.inpt_hid1 = nn.Linear(config.input, config.hid)
        self.hid1_hid1 = nn.Linear(config.hid, config.hid) # A1
        self.hid1_hid2 = nn.Linear(config.hid, config.hid)
        self.hid2_hid2 = nn.Linear(config.hid, config.hid) # A2
        self.hid2_out = nn.Linear(config.hid, config.output)
        if config.small_init:
            self.hid1_hid1.weight.data = 0.2 * self.hid1_hid1.weight.data
            self.hid2_hid2.weight.data = 0.2 * self.hid2_hid2.weight.data
        
        nn.init.orthogonal_(self.hid1_hid1.weight)  # 主要用以解决深度网络的梯度消失爆炸问题，在RNN中经常使用
        nn.init.orthogonal_(self.hid2_hid2.weight)
        nn.init.xavier_uniform_(self.inpt_hid1.weight) # 保持输入输出的方差一致，避免所有输出值都趋向于0。通用方法，适用于任何激活函数
        nn.init.xavier_uniform_(self.hid1_hid2.weight)
        nn.init.xavier_uniform_(self.hid2_out.weight)
        
        nn.init.constant_(self.inpt_hid1.bias, 0)
        nn.init.constant_(self.hid1_hid2.bias, 0)
        nn.init.constant_(self.hid1_hid1.bias, 0)
        nn.init.constant_(self.hid2_hid2.bias, 0)
        
        self.tau_adp_h1 = nn.Parameter(torch.Tensor(config.hid))
        self.tau_adp_h2 = nn.Parameter(torch.Tensor(config.hid))
        self.tau_adp_o = nn.Parameter(torch.Tensor(config.output))
        self.tau_m_h1 = nn.Parameter(torch.Tensor(config.hid))
        self.tau_m_h2 = nn.Parameter(torch.Tensor(config.hid))
        self.tau_m_o = nn.Parameter(torch.Tensor(config.output))

        nn.init.normal_(self.tau_adp_h1, 150, 10)
        nn.init.normal_(self.tau_adp_h2, 150, 10)
        nn.init.normal_(self.tau_adp_o, 150, 10)
        nn.init.normal_(self.tau_m_h1, 20., 5)
        nn.init.normal_(self.tau_m_h2, 20., 5)
        nn.init.normal_(self.tau_m_o, 20., 5)

        self.b_hid1 = self.b_hid2 = self.b_out = 0
        
        if not config.input_learn:
            for name, p in self.named_parameters():
                if 'inpt' in name:
                    p.requires_grad = False
    
    def output_Neuron(self, inputs, mem, tau_m, dt=1):
        """The read out neuron is leaky integrator without spike"""
        # alpha = torch.exp(-1. * dt / torch.FloatTensor([30.])).to(config.device)
        alpha = torch.exp(-1. * dt / tau_m).to(config.device)
        mem = mem * alpha + (1. - alpha) * config.R_m * inputs
        return mem
    
    def mem_update_adp(self, inputs, mem, spike, tau_adp, b, tau_m, dt=1, isAdapt=1):
        alpha = torch.exp(-1. * dt / tau_m).to(config.device)
        ro = torch.exp(-1. * dt / tau_adp).to(config.device)
        if isAdapt: beta = 1.8
        else:       beta = 0.
        b = ro * b + (1 - ro) * spike
        B = config.b_j0 + beta * b
        mem = mem * alpha + (1 - alpha) * config.R_m * inputs - B * spike * dt
        spike = act_fun_adp(mem - B)
        return mem, spike, B, b
    
    def forward(self, input, mask):
        batch, nb_steps, _ = input.shape
        self.b_hid1 = self.b_hid2 = self.b_out = config.b_j0
        
        hid1_mem = torch.rand(batch, config.hid).to(config.device)
        hid1_spk = torch.zeros(batch, config.hid).to(config.device)
        
        hid2_mem = torch.rand(batch, config.hid).to(config.device)
        hid2_spk = torch.zeros(batch, config.hid).to(config.device)
        
        out_mem = torch.rand(batch, config.output).to(config.device)
        output = torch.zeros(batch, config.output).to(config.device)
        
        sum1_spk = torch.zeros(batch, config.hid).to(config.device)
        sum2_spk = torch.zeros(batch, config.hid).to(config.device)
        
        if config.dropout > 0:
            self.hid1_hid1.weight.data = self.hid1_hid1.weight.data * mask[0].T
            self.hid2_hid2.weight.data = self.hid2_hid2.weight.data * mask[1].T
        for t in range(nb_steps):
            input_t = input[:,t,:].float()
            ########## Layer 1 ##########
            inpt_hid1 = self.inpt_hid1(input_t) + self.hid1_hid1(hid1_spk)
            hid1_mem, hid1_spk, theta_h1, self.b_hid1 = self.mem_update_adp(inpt_hid1, hid1_mem, hid1_spk, self.tau_adp_h1, self.b_hid1, self.tau_m_h1)
            sum1_spk += hid1_spk
            # hid1_spk = self.dp(hid1_spk)
            
            ########## Layer 2 ##########
            inpt_hid2 = self.hid1_hid2(hid1_spk) + self.hid2_hid2(hid2_spk)
            hid2_mem, hid2_spk, theta_h2, self.b_hid2 = self.mem_update_adp(inpt_hid2, hid2_mem, hid2_spk, self.tau_adp_h2, self.b_hid2, self.tau_m_h2)
            sum2_spk += hid2_spk
            # hid2_spk = self.dp(hid2_spk)
            
            ########## Layer out ########
            inpt_out = self.hid2_out(hid2_spk)
            out_mem = self.output_Neuron(inpt_out, out_mem, self.tau_m_o)
            if t > 10:
                output += F.softmax(out_mem, dim=1)
            
        sum1_spk /= nb_steps
        sum2_spk /= nb_steps

        A_norm = torch.norm(self.hid1_hid1.weight, p=1) + torch.norm(self.hid2_hid2.weight, p=1)
        # cluster_in1 = 0 # 簇内聚类程度
        # cluster_out1 = 0
        # global_mean1 = 0 # 全局中心位置
        # cluster_in2 = 0 # 簇内聚类程度
        # cluster_out2 = 0
        # global_mean2 = 0 # 全局中心位置
        # for i in range(config.output):
        #     center1 = self.hid1_hid1.weight[i*20:(i+1)*20].mean(0)
        #     cluster_in1 += ((self.hid1_hid1.weight[i*20:(i+1)*20] - center1)**2).mean()
        #     global_mean1 += 0.1*center1
        #     center2 = self.hid2_hid2.weight[i*20:(i+1)*20].mean(0)
        #     cluster_in2 += ((self.hid2_hid2.weight[i*20:(i+1)*20] - center2)**2).mean()
        #     global_mean2 += 0.1*center2
        # for i in range(config.output):
        #     cluster_out1 += ((global_mean1 - self.hid1_hid1.weight[i*20:(i+1)*20].mean(0))**2).mean()
        #     cluster_out2 += ((global_mean2 - self.hid2_hid2.weight[i*20:(i+1)*20].mean(0))**2).mean()

        return output, sum1_spk, sum2_spk, A_norm, 0,0 # cluster_in1+cluster_in2, cluster_out1+cluster_out2

#####################################
########### load SHD data ###########
def load_shd_or_ssc():
    train_X = np.load('data/trainX_4ms.npy')
    train_y = np.load('data/trainY_4ms.npy').astype(float)

    test_X = np.load('data/testX_4ms.npy')
    test_y = np.load('data/testY_4ms.npy').astype(float)

    print('dataset shape: ', train_X.shape)
    print('dataset shape: ', test_X.shape)

    tensor_trainX = torch.Tensor(train_X)  # transform to torch tensor
    tensor_trainY = torch.Tensor(train_y)
    train_dataset = torch.utils.data.TensorDataset(tensor_trainX, tensor_trainY)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch, shuffle=True)
    tensor_testX = torch.Tensor(test_X)  # transform to torch tensor
    tensor_testY = torch.Tensor(test_y)
    test_dataset = torch.utils.data.TensorDataset(tensor_testX, tensor_testY)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.batch, shuffle=False)
    return train_loader, test_loader

################################################
########### define training pipeline ###########
class Experiment:
    """Training and testing models on SHD and SSC datasets."""
    def __init__(self):
        # Initialize logging
        os.makedirs(config.save_dir) if not os.path.exists(config.save_dir) else None
        logging.FileHandler(filename=config.save_dir + "/exp.log", mode="a", encoding=None, delay=False,)
        logging.basicConfig(filename=config.save_dir + "/exp.log", level=logging.INFO, format="%(message)s", filemode='a',)
        logging.info("===== Exp configuration =====")
        for var in vars(config):
            if var[0] != '_': 
                logging.info(str(var) + ':\t\t' + str(vars(config)[var]))
        
        self.train_loader, self.test_loader = load_shd_or_ssc()
        self.net = RC().to(config.device)
        logging.info(f"\nCreated new spiking model:\n {self.net}\n")
        self.nb_params = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
        logging.info(f"Total number of trainable parameters is {self.nb_params}")
        
        # optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config.lr, weight_decay=config.weight_decay)
        # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.lr)
        # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.lr, eps=1e-5)
        self.optimizer = torch.optim.Adam(self.net.parameters(), config.lr)
        # self.scheduler = StepLR(self.optimizer, step_size=10, gamma=.5)
        self.scheduler = ReduceLROnPlateau(
            optimizer=self.optimizer,
            mode="max",
            factor=config.scheduler_factor,
            patience=config.scheduler_patience,
            min_lr=1e-6,
        )
        self.loss_fn = nn.CrossEntropyLoss()
        
    def train_one_epoch(self, e, mask):
        start = time.time()
        self.net.train()
        losses, epoch_spike_rate = [], 0
        correct, total = 0, 0

        for step, (images, labels) in enumerate(self.train_loader):
            images = images.view(-1, config.nb_steps, config.input).requires_grad_().to(config.device)
            labels = labels.long().to(config.device)
            
            outputs, sum1_spk, sum2_spk, A_norm, cluster_in, cluster_out = self.net(images, mask)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted.cpu() == labels.long().cpu()).sum()
            
            if config.smoothing>0:
                with torch.no_grad():
                    true_dist = torch.zeros_like(outputs)
                    true_dist.fill_(config.smoothing / (config.output - 1))
                    true_dist.scatter_(1, labels.unsqueeze(1), 1.0 - config.smoothing)
                loss = self.loss_fn(outputs, true_dist)
            else: loss = self.loss_fn(outputs, labels)
            losses.append(loss.item())

            # Spike activity
            firing_rates = sum1_spk.mean()*0.5 + sum2_spk.mean()*0.5
            epoch_spike_rate += torch.mean(firing_rates)
            reg_quiet = F.relu(config.fr_min - firing_rates).sum()
            reg_burst = F.relu(firing_rates - config.fr_max).sum()
            loss += config.fr_norm * (reg_quiet + reg_burst)
            # loss += config.l1 * F.relu(A_norm - config.l1_targ) # , torch.max(A_norm-6000, 0) 规定一个区间，
            # loss += (0.02*cluster_in - 0.04*cluster_out)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted.cpu() == labels.long().cpu()).sum()
        
        # Learning rate of whole epoch
        current_lr = self.optimizer.param_groups[-1]["lr"]
        tr_loss = np.mean(losses)
        tr_acc = 100. * correct.numpy() / total
        epoch_spike_rate /= step
        elapsed = str(timedelta(seconds=time.time() - start))
        logging.info(f'Epoch:{e}, Tr Loss:{tr_loss}, Acc:{tr_acc}, Time:{elapsed},\tA Norm:{A_norm.item()},\tFr:{epoch_spike_rate}, \
                     lr:{current_lr}, Mask:{(mask[0]==0).sum().item()/config.hid**2}, Cin:{cluster_in}, Cout:{cluster_out}')
        return tr_acc

    def valid_one_epoch(self, trial, e, best_epoch, best_acc, mask):
        with torch.no_grad():
            self.net.eval()
            losses, epoch_spike_rate, correct, total = [], 0, 0, 0
            for step, (images, labels) in enumerate(self.test_loader):
                images = images.view(-1, config.nb_steps, config.input).to(config.device)
                labels = labels.long().to(config.device)
                if config.noise_test > 0:
                    images += torch.rand_like(images) * config.noise_test
                outputs, sum1_spk, sum2_spk, _, _, _ = self.net(images, mask)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted.cpu() == labels.long().cpu()).sum()
                loss = self.loss_fn(outputs, labels)
                losses.append(loss.item())
                firing_rates = sum1_spk.mean()*0.5 + sum2_spk.mean()*0.5
                epoch_spike_rate += torch.mean(firing_rates)
            valid_acc = 100. * correct.numpy() / total
            valid_loss = np.mean(losses)
            epoch_spike_rate /= step
            logging.info(f"Epoch {e}: Te loss:{valid_loss}, acc:{valid_acc}, Fr:{epoch_spike_rate}")

            self.scheduler.step(valid_acc)
            # Update best epoch and accuracy
            if valid_acc > best_acc:
                best_acc = valid_acc
                best_epoch = e
                torch.save([self.net.state_dict(), mask], f"{config.save_dir}/best_model.tar")
                logging.info(f"\nBest model saved with valid acc={valid_acc}")
            if e % config.ckpt_freq==0:
                torch.save([self.net.state_dict(), mask], config.save_dir+'/model-{:d}-{:d}-{:.4f}.tar'.format(trial, e, valid_acc))
            logging.info("\n-----------------------------\n")
            return best_epoch, best_acc, valid_acc
        
    def forward(self, trial):
        self.net = RC().to(config.device)
        best_epoch, best_acc = 0, 0
        train_accs, test_accs = [], [] # for error bar
        
        #### M1 mask
        a = torch.zeros((config.hid, config.hid), dtype=torch.int)
        for i in range(4): a[i*32:(i+1)*32, i*32:(i+1)*32] = 1.
        invalid_zeros = 1-(a==1).sum().item()/config.hid**2
        if invalid_zeros < config.dropout:
            b = (torch.rand(config.hid, config.hid) > (config.dropout-invalid_zeros)/(1-invalid_zeros)).int() * (1-torch.eye(config.hid, config.hid, dtype=int))
            m1 = a & b
            m1 += torch.eye(config.hid, config.hid, dtype=int)
        else: m1 = a
        
        #### M2 mask
        a = torch.zeros((config.hid, config.hid), dtype=torch.int)
        for i in range(4): a[i*32:(i+1)*32, i*32:(i+1)*32] = 1.
        invalid_zeros = 1-(a==1).sum().item()/config.hid**2
        if invalid_zeros < config.dropout:
            b = (torch.rand(config.hid, config.hid) > (config.dropout-invalid_zeros)/(1-invalid_zeros)).int() * (1-torch.eye(config.hid, config.hid, dtype=int))
            m2 = a & b
            m2 += torch.eye(config.hid, config.hid, dtype=int)
        else: m2 = a
        # m1 = (torch.rand(config.hid, config.hid) > config.dropout).int() * (1-torch.eye(config.hid, config.hid)).int()
        # m2 = (torch.rand(config.hid, config.hid) > config.dropout).int() * (1-torch.eye(config.hid, config.hid)).int()
        mask = [m1.float().to(config.device), m2.float().to(config.device)]
        torch.save([self.net.state_dict(), mask], config.save_dir+'/before-train-{:d}-{:d}-{:.4f}.tar'.format(trial, 0, 0))
        
        logging.info(f"\n------ Trial {trial} begin ------\n")
        for e in range(best_epoch + 1, best_epoch + config.epoch + 1):
            train_acc = self.train_one_epoch(e, mask)
            best_epoch, best_acc, test_acc = self.valid_one_epoch(trial, e, best_epoch, best_acc, mask)
            train_accs.append(train_acc); test_accs.append(test_acc)

            if (m1==0).sum().item()/config.hid**2 <= config.dropout_stop or \
                (m2==0).sum().item()/config.hid**2 <= config.dropout_stop:
                m1 = m1&((torch.rand(config.hid, config.hid) > config.dropout_stepping).int() * (1-torch.eye(config.hid, config.hid)).int())
                m2 = m2&((torch.rand(config.hid, config.hid) > config.dropout_stepping).int() * (1-torch.eye(config.hid, config.hid)).int())
                mask = [m1.float().to(config.device), m2.float().to(config.device)]
        
        logging.info(f"\nBest valid acc at epoch {best_epoch}: {best_acc}\n")
        logging.info(f"\n------ Trial {trial} finished ------\n")
        return np.array(train_accs), np.array(test_accs)
    
    def multiple_trial(self,):
        train_acc_log = np.zeros((config.epoch, config.trials))
        test_acc_log = np.zeros((config.epoch, config.trials))
        for i in range(config.trials):
            np.random.seed(config.seed+i)
            torch.manual_seed(config.seed+i)
            torch.cuda.manual_seed_all(config.seed+i)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

            train_acc, test_acc = self.forward(i)
            train_acc_log[:,i] = train_acc
            test_acc_log[:,i] = test_acc
        return train_acc_log, test_acc_log
    
    def plot_errorbar(self, train_acc_log, test_acc_log, file_name):
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

        plt.plot(list(range(config.epoch)), train_mean, color='deeppink', label='train')
        plt.fill_between(list(range(config.epoch)), train_mean-train_std, train_mean+train_std, color='deeppink', alpha=0.2)
        # plt.fill_between(list(range(config.epoch)), train_min, train_max, color='violet', alpha=0.2)

        plt.plot(list(range(config.epoch)), test_mean, color='blue', label='test')
        plt.fill_between(list(range(config.epoch)), test_mean-test_std, test_mean+test_std, color='blue', alpha=0.2)
        # plt.fill_between(list(range(config.epoch)), test_min, test_max, color='blue', alpha=0.2)

        plt.legend()
        plt.grid()
        # plt.axis([-5, 105, 75, 95])
        plt.savefig(file_name)
        # plt.show()

def test(model, dataloader, mask):
    
    with torch.no_grad():
        model.eval()
        correct, total = 0, 0
        for images, labels in dataloader:
            images = images.view(-1, config.nb_steps, config.input).to(config.device)
            if config.noise_test > 0:
                images += torch.rand_like(images) * config.noise_test
            outputs, _, _, _, _, _ = model(images, mask)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted.cpu() == labels.long().cpu()).sum()
        accuracy = 100. * correct.numpy() / total
    return accuracy

def multiple_trial():
    train_acc_log = np.zeros((config.epoch, config.trials))
    test_acc_log = np.zeros((config.epoch, config.trials))
    for i in range(config.trials):
        print('************** Trial ', i, ' **************')
        np.random.seed(config.seed+i)
        torch.manual_seed(config.seed+i)
        torch.cuda.manual_seed_all(config.seed+i)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        model = RC().to(config.device)
        criterion = nn.CrossEntropyLoss()
        # optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config.lr, weight_decay=config.weight_decay)
        # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.lr)
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.lr, eps=1e-5)
        train_acc, test_acc = train(i, model, optimizer, criterion, config.epoch, train_loader, test_loader)
        train_acc_log[:,i] = train_acc
        test_acc_log[:,i] = test_acc
    return train_acc_log, test_acc_log

###################################
########### start train ###########
if __name__ == "__main__":
    exp = Experiment()
    train_acc_log, test_acc_log = exp.multiple_trial()
    exp.plot_errorbar(train_acc_log, test_acc_log, './fig/'+ config.date +'.pdf')
    