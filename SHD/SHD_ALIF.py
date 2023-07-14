'''
modified from https://github.com/byin-cwi/Efficient-spiking-networks/blob/main/SHD/SHD_2layer_ALIF.py

'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import time
from torch.optim.lr_scheduler import StepLR

class config:
    seed = 2
    input = 700
    hid = 128         # number of RC Neurons
    output = 20
    time_step = 250  # Number of steps to unroll
    lr = 1e-2
    b_j0 = 0.01  # neural threshold baseline
    R_m = 1 # membrane resistance
    dt = 1
    batch = 256
    gamma = 0.5  # gradient scale
    gradient_type = 'MG' # 'G', 'slayer', 'linear' 窗型函数
    scale = 6.        # special for 'MG'
    hight = 0.15      # special for 'MG'
    lens = 0.5  # hyper-parameters of approximate function
    epoch = 40
    trials = 5
    dropout = 0
    dropout_stepping = 0.01
    dropout_stop = 0.90
    smoothing = 0.1
    device = torch.device('cuda')

np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.cuda.manual_seed_all(config.seed)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

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

def gaussian(x, mu=0., sigma=.5):
    return torch.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / torch.sqrt(2 * torch.tensor(torch.pi)) / sigma

class ActFun_adp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):  # input = membrane potential- threshold
        ctx.save_for_backward(input)
        return input.gt(0).float()  # is firing ???

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

class SRNN(nn.Module):
    def __init__(self) -> None:
        super(SRNN, self).__init__()
        input = config.input
        hid = config.hid
        out = config.output
        self.inpt_hid1 = nn.Linear(input, hid)
        self.hid1_hid1 = nn.Linear(hid, hid) # A1
        self.hid1_hid2 = nn.Linear(hid, hid)
        self.hid2_hid2 = nn.Linear(hid, hid) # A2
        self.hid2_hid3 = nn.Linear(hid, hid)
        self.hid3_hid3 = nn.Linear(hid, hid) # A3
        self.hid3_out = nn.Linear(hid, out)
        
        # self.hid1_hid1.weight.data = 0.2 * self.hid1_hid1.weight.data
        # self.hid2_hid2.weight.data = 0.2 * self.hid2_hid2.weight.data
        
        nn.init.orthogonal_(self.hid1_hid1.weight)  # 主要用以解决深度网络的梯度消失爆炸问题，在RNN中经常使用
        nn.init.orthogonal_(self.hid2_hid2.weight)
        nn.init.orthogonal_(self.hid3_hid3.weight)
        nn.init.xavier_uniform_(self.inpt_hid1.weight) # 保持输入输出的方差一致，避免所有输出值都趋向于0。通用方法，适用于任何激活函数
        nn.init.xavier_uniform_(self.hid1_hid2.weight)
        nn.init.xavier_uniform_(self.hid2_hid3.weight)
        nn.init.xavier_uniform_(self.hid3_out.weight)
        
        # nn.init.constant_(self.inpt_hid1.bias, 0)
        # nn.init.constant_(self.hid1_hid2.bias, 0)
        # nn.init.constant_(self.hid1_hid1.bias, 0)
        # nn.init.constant_(self.hid2_hid2.bias, 0)
        
        self.thr1 = nn.Parameter(torch.rand(config.hid), requires_grad=True)
        self.decay1 = nn.Parameter(torch.rand(config.hid)*config.decay, requires_grad=True)
        self.rst1 = nn.Parameter(torch.rand(config.hid)*config.rst, requires_grad=True)
        
        self.thr2 = nn.Parameter(torch.rand(config.hid), requires_grad=True)
        self.decay2 = nn.Parameter(torch.rand(config.hid)*config.decay, requires_grad=True)
        self.rst2 = nn.Parameter(torch.rand(config.hid)*config.rst, requires_grad=True)
        
        self.thr3 = nn.Parameter(torch.rand(config.hid), requires_grad=True)
        self.decay3 = nn.Parameter(torch.rand(config.hid)*config.decay, requires_grad=True)
        self.rst3 = nn.Parameter(torch.rand(config.hid)*config.rst, requires_grad=True)
        
        self.alpha = nn.Parameter(torch.rand(config.output), requires_grad=True)
    
    def output_Neuron(self, inputs, mem, alpha):
        """
        The read out neuron is leaky integrator without spike
        """
        mem = mem * alpha + (1. - alpha) * config.R_m * inputs
        return mem
    
    def mem_update(self, input, mem, spk, thr, decay, rst):
        mem = rst * spk + mem * decay * (1-spk) + input
        spike = act_fun_adp(mem - thr)
        return mem, spike
    
    def forward(self, input, mask, device='cuda'):
        A1_mask, A2_mask, A3_mask, _ = mask
        batch = input.shape[0]
        time_step = input.shape[1]
        
        hid1_mem = torch.rand(batch, config.hid).to(device)
        # hid1_mem = torch.zeros(batch, config.hid).uniform_(0, 0.1).to(device)
        hid1_spk = torch.zeros(batch, config.hid).to(device)
        
        hid2_mem = torch.rand(batch, config.hid).to(device)
        # hid2_mem = torch.zeros(batch, config.hid).uniform_(0, 0.1).to(device)
        hid2_spk = torch.zeros(batch, config.hid).to(device)
        
        hid3_mem = torch.rand(batch, config.hid).to(device)
        # hid2_mem = torch.zeros(batch, config.hid).uniform_(0, 0.1).to(device)
        hid3_spk = torch.zeros(batch, config.hid).to(device)
        
        out_mem = torch.rand(batch, config.output).to(device)
        # out_mem = torch.zeros(batch, config.output).uniform_(0, 0.1).to(device)
        output = torch.zeros(batch, config.output).to(device)
        
        sum1_spk = torch.zeros(batch, config.hid).to(device)
        sum2_spk = torch.zeros(batch, config.hid).to(device)
        sum3_spk = torch.zeros(batch, config.hid).to(device)
        
        if config.dropout>0:
            self.hid1_hid1.weight.data = self.hid1_hid1.weight.data * A1_mask.T.to(device)
            self.hid2_hid2.weight.data = self.hid2_hid2.weight.data * A2_mask.T.to(device)
            self.hid3_hid3.weight.data = self.hid3_hid3.weight.data * A3_mask.T.to(device)
        for t in range(time_step):
            input_t = input[:,t,:].float()
            
            ########## Layer 1 ##########
            inpt_hid1 = self.inpt_hid1(input_t) + self.hid1_hid1(hid1_spk)
            hid1_mem, hid1_spk = self.mem_update(inpt_hid1, hid1_mem, hid1_spk, self.thr1, self.decay1, self.rst1)
            sum1_spk += hid1_spk
            
            ########## Layer 2 ##########
            inpt_hid2 = self.hid1_hid2(hid1_spk) + self.hid2_hid2(hid2_spk)
            hid2_mem, hid2_spk = self.mem_update(inpt_hid2, hid2_mem, hid2_spk, self.thr2, self.decay2, self.rst2)
            sum2_spk += hid2_spk
            
            ########## Layer 3 ##########
            inpt_hid3 = self.hid2_hid3(hid2_spk) + self.hid3_hid3(hid3_spk)
            hid3_mem, hid3_spk = self.mem_update(inpt_hid3, hid3_mem, hid3_spk, self.thr3, self.decay3, self.rst3)
            sum3_spk += hid3_spk
            
            ########## Layer out ########
            inpt_out = self.hid3_out(hid3_spk)
            out_mem = self.output_Neuron(inpt_out, out_mem, self.alpha)
            if t > 10:
                output += F.softmax(out_mem, dim=1)
            
        sum1_spk /= time_step
        sum2_spk /= time_step

        A_norm = torch.norm(self.hid1_hid1.weight, p=1) + \
                 torch.norm(self.hid2_hid2.weight, p=1) + \
                 torch.norm(self.hid3_hid3.weight, p=1)
        return output, sum1_spk, sum2_spk, A_norm

class SRNN_custom(nn.Module):
    def __init__(self):
        super(SRNN_custom, self).__init__()
        input = config.input
        hid = config.hid
        out = config.output
        self.inpt_hid1 = nn.Linear(input, hid)
        self.hid1_hid1 = nn.Linear(hid, hid) # A1
        self.hid1_hid2 = nn.Linear(hid, hid)
        self.hid2_hid2 = nn.Linear(hid, hid) # A2
        self.hid2_out = nn.Linear(hid, out)
        
        # self.hid1_hid1.weight.data = 0.2 * self.hid1_hid1.weight.data
        # self.hid2_hid2.weight.data = 0.2 * self.hid2_hid2.weight.data
        
        nn.init.orthogonal_(self.hid1_hid1.weight)  # 主要用以解决深度网络的梯度消失爆炸问题，在RNN中经常使用
        nn.init.orthogonal_(self.hid2_hid2.weight)
        nn.init.xavier_uniform_(self.inpt_hid1.weight) # 保持输入输出的方差一致，避免所有输出值都趋向于0。通用方法，适用于任何激活函数
        nn.init.xavier_uniform_(self.hid1_hid2.weight)
        nn.init.xavier_uniform_(self.hid2_out.weight)

        nn.init.constant_(self.inpt_hid1.bias, 0)
        nn.init.constant_(self.hid1_hid2.bias, 0)
        nn.init.constant_(self.hid1_hid1.bias, 0)
        nn.init.constant_(self.hid2_hid2.bias, 0)
        
        self.tau_adp_h1 = nn.Parameter(torch.Tensor(hid))
        self.tau_adp_h2 = nn.Parameter(torch.Tensor(hid))
        self.tau_adp_o = nn.Parameter(torch.Tensor(out))
        self.tau_m_h1 = nn.Parameter(torch.Tensor(hid))
        self.tau_m_h2 = nn.Parameter(torch.Tensor(hid))
        self.tau_m_o = nn.Parameter(torch.Tensor(out))

        nn.init.normal_(self.tau_adp_h1, 150, 10)
        nn.init.normal_(self.tau_adp_h2, 150, 10)
        nn.init.normal_(self.tau_adp_o, 150, 10)
        nn.init.normal_(self.tau_m_h1, 20., 5)
        nn.init.normal_(self.tau_m_h2, 20., 5)
        nn.init.normal_(self.tau_m_o, 20., 5)

        self.dp = nn.Dropout(0.1)

        self.b_hid1 = self.b_hid2 = self.b_o = 0
    
    def output_Neuron(self, inputs, mem, tau_m, dt=1):
        """
        The read out neuron is leaky integrator without spike
        """
        # alpha = torch.exp(-1. * dt / torch.FloatTensor([30.])).cuda()
        alpha = torch.exp(-1. * dt / tau_m).cuda()
        mem = mem * alpha + (1. - alpha) * config.R_m * inputs
        return mem
    
    def mem_update_adp(self, inputs, mem, spike, tau_adp, b, tau_m, dt=1, isAdapt=1):
        alpha = torch.exp(-1. * dt / tau_m).cuda()
        ro = torch.exp(-1. * dt / tau_adp).cuda()
        if isAdapt: beta = 1.8
        else:       beta = 0.
        b = ro * b + (1 - ro) * spike
        B = config.b_j0 + beta * b
        mem = mem * alpha + (1 - alpha) * config.R_m * inputs - B * spike * dt
        spike = act_fun_adp(mem - B)
        return mem, spike, B, b
    
    def forward(self, input, mask, device='cuda'):
        A1_mask, A2_mask = mask
        batch, time_step, _ = input.shape
        self.b_hid1 = self.b_hid2 = self.b_o = config.b_j0
        hid1_mem = torch.rand(batch, config.hid).to(device)
        hid1_spk = torch.zeros(batch, config.hid).to(device)
        hid2_mem = torch.rand(batch, config.hid).to(device)
        hid2_spk = torch.zeros(batch, config.hid).to(device)
        out_mem = torch.rand(batch, config.output).to(device)
        output = torch.zeros(batch, config.output).to(device)

        hidden_spike_ = []
        hidden_mem_ = []
        h2o_mem_ = []
        if config.dropout > 0:
            self.hid1_hid1.weight.data = self.hid1_hid1.weight.data * A1_mask.T.to(device)
            self.hid2_hid2.weight.data = self.hid2_hid2.weight.data * A2_mask.T.to(device)
        for t in range(time_step):
            input_t = input[:,t,:].float()
            ########## Layer 1 ##########
            inpt_hid1 = self.inpt_hid1(input_t) + self.hid1_hid1(hid1_spk)
            hid1_mem, hid1_spk, theta_h1, self.b_hid1 = self.mem_update_adp(inpt_hid1, hid1_mem, hid1_spk,
                                                                          self.tau_adp_h1, self.b_hid1, self.tau_m_h1)
            # hid1_spk = self.dp(hid1_spk)
            ########## Layer 2 ##########
            inpt_hid2 = self.hid1_hid2(hid1_spk) + self.hid2_hid2(hid2_spk)
            hid2_mem, hid2_spk, theta_h2, self.b_hid2 = self.mem_update_adp(inpt_hid2, hid2_mem, hid2_spk,
                                                                          self.tau_adp_h2, self.b_hid2, self.tau_m_h2)
            ########## Layer out ########
            inpt_out = self.hid2_out(hid2_spk)
            out_mem = self.output_Neuron(inpt_out, out_mem, self.tau_m_o)
            if t > 10:
                output += F.softmax(out_mem, dim=1)

            hidden_spike_.append(hid2_spk.data.cpu().numpy())
            hidden_mem_.append(hid2_mem.data.cpu().numpy())
            h2o_mem_.append(out_mem.data.cpu().numpy())
        A_norm = torch.norm(self.hid1_hid1.weight, p=1) + torch.norm(self.hid2_hid2.weight, p=1)
        return output, hidden_spike_, hidden_mem_, h2o_mem_



def train(model, optimizer, criterion, num_epochs, train_loader, test_loader, device):
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    train_accs, test_accs = [], []
    best_accuracy = 85
    m1 = (torch.rand(config.hid, config.hid) > config.dropout).int() * (1-torch.eye(config.hid, config.hid)).int()
    m2 = (torch.rand(config.hid, config.hid) > config.dropout).int() * (1-torch.eye(config.hid, config.hid)).int()
    mask = [m1.float(), m2.float()]
    
    for epoch in range(num_epochs):
        now = time.time()
        loss_sum = 0
        correct, total = 0, 0
        for i, (images, labels) in enumerate(train_loader):
            images = images.view(-1, config.time_step, config.input).requires_grad_().to(device)
            labels = labels.long().to(device)
            
            optimizer.zero_grad()
            outputs, _, _, _ = model(images.to(device), mask, device)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted.cpu() == labels.long().cpu()).sum()
            
            # label smoothing
            if config.smoothing>0:
                with torch.no_grad():
                    true_dist = torch.zeros_like(outputs)
                    true_dist.fill_(config.smoothing / (config.output - 1))
                    true_dist.scatter_(1, labels.unsqueeze(1), 1.0 - config.smoothing)
                
                loss = criterion(outputs, true_dist)
            else:
                loss = criterion(outputs, labels)
            loss_sum += loss.item()
            loss.backward()
            optimizer.step()
        scheduler.step()
        tr_acc = 100. * correct.numpy() / total
        ts_acc, fr = test(model, test_loader, mask, is_test=0)
        if ts_acc > best_accuracy and tr_acc > 85:
            torch.save(model, './model_no_mask-label_smooth_0.2_' + str(ts_acc) + '-readout-2layer-v2-4ms.pth')
            best_accuracy = ts_acc

        print('epoch: ', epoch, 
              '. Loss: ', loss.item(), 
              '. Tr Accuracy: ', tr_acc, 
              '. Ts Accuracy: ', ts_acc, 
              '. Fr: ', fr,
              '. Time: ', time.time()-now
              )

        train_accs.append(tr_acc)
        test_accs.append(ts_acc)
        if (m1==0).sum().item()/config.hid**2 <= config.dropout_stop or \
            (m2==0).sum().item()/config.hid**2 <= config.dropout_stop:
            m1 = m1&((torch.rand(config.hid, config.hid) > config.dropout_stepping).int() * (1-torch.eye(config.hid, config.hid)).int())
            m2 = m2&((torch.rand(config.hid, config.hid) > config.dropout_stepping).int() * (1-torch.eye(config.hid, config.hid)).int())
            mask = [m1.float(), m2.float()]
    return np.array(train_accs), np.array(test_accs)

def test(model, dataloader, mask, is_test=0):
    with torch.no_grad():
        correct, total = 0, 0
        for images, labels in dataloader:
            images = images.view(-1, config.time_step, config.input).to(config.device)
            outputs, fr_, _, _ = model(images, mask)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted.cpu() == labels.long().cpu()).sum()
        accuracy = 100. * correct.numpy() / total
        if is_test:
            print('Mean FR: ', np.array(fr_).mean())
    return accuracy, np.array(fr_).mean()


###############################
model = SRNN_custom().to(config.device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, eps=1e-5)
# acc = train(model, optimizer, criterion, config.epoch, train_loader, test_loader, config.device)
# test_acc, fr = test(model, test_loader, is_test=1)
# print(' Accuracy: ', test_acc)

# base_params = [model.inpt_hid1.weight, model.inpt_hid1.bias,
#                model.hid1_hid1.weight, model.hid1_hid1.bias,
#                model.hid1_hid2.weight, model.hid1_hid2.bias,
#                model.hid2_hid2.weight, model.hid2_hid2.bias,
#                model.hid2_out.weight, model.hid2_out.bias]
# optimizer = torch.optim.Adam([
#     {'params': base_params},
#     {'params': model.tau_adp_h1, 'lr': learning_rate * 5},
#     {'params': model.tau_adp_h2, 'lr': learning_rate * 5},
#     {'params': model.tau_m_h1, 'lr': learning_rate * 1},
#     {'params': model.tau_m_h2, 'lr': learning_rate * 1},
#     {'params': model.tau_m_o, 'lr': learning_rate * 1}],
#     lr=learning_rate, eps=1e-5)




def multiple_trial():
    train_acc_log = np.zeros((config.epoch, config.trials))
    test_acc_log = np.zeros((config.epoch, config.trials))
    for i in range(config.trials):
        print('************** ', i, ' **************')
        np.random.seed(config.seed+i)
        torch.manual_seed(config.seed+i)
        torch.cuda.manual_seed_all(config.seed+i)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # model = RC().to(config.device)
        model = SRNN_custom().to(config.device)
        criterion = nn.CrossEntropyLoss()
        # optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config.lr, weight_decay=config.weight_decay)
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.lr, eps=1e-5)
        train_acc, test_acc = train(model, optimizer, criterion, config.epoch, train_loader, test_loader, config.device)
        train_acc_log[:,i] = train_acc
        test_acc_log[:,i] = test_acc
    return train_acc_log, test_acc_log

def plot_errorbar(train_acc_log, test_acc_log, file_name):
    train_mean = np.mean(train_acc_log, axis=1)
    train_std = np.std(train_acc_log, axis=1)
    train_var = np.var(train_acc_log, axis=1)
    train_max = np.max(train_acc_log, axis=1)
    train_min = np.min(train_acc_log, axis=1)

    test_mean = np.mean(test_acc_log, axis=1)
    test_std = np.std(test_acc_log, axis=1)
    test_var = np.var(test_acc_log, axis=1)
    test_max = np.max(test_acc_log, axis=1)
    test_min = np.min(test_acc_log, axis=1)

    plt.plot(list(range(config.epoch)), train_mean, color='deeppink', label='train mean')
    plt.fill_between(list(range(config.epoch)), train_mean-train_std, train_mean+train_std, color='deeppink', alpha=0.2)
    # plt.fill_between(list(range(config.epoch)), train_min, train_max, color='violet', alpha=0.2)

    plt.plot(list(range(config.epoch)), test_mean, color='blue', label='test mean')
    plt.fill_between(list(range(config.epoch)), test_mean-test_std, test_mean+test_std, color='blue', alpha=0.2)
    # plt.fill_between(list(range(config.epoch)), test_min, test_max, color='blue', alpha=0.2)

    plt.legend()
    plt.grid()
    # plt.axis([-5, 105, 75, 95])
    plt.savefig(file_name)
    # plt.show()

train_acc_log, test_acc_log = multiple_trial()
plot_errorbar(train_acc_log, test_acc_log, 'SHD_ALIF_no_mask-label_smooth_0.3.pdf')


# dataset shape:  (8156, 250, 700)
# dataset shape:  (2264, 250, 700)
# gradient_type:  MG
# hight:  0.15 ;scale:  6.0
# device: cuda:0
# epoch:  1 . Loss:  1.4047224521636963 . Tr Accuracy:  30.946542422756252 . Ts Accuracy:  40.90106007067138 Fr:  0.10361849
# epoch:  2 . Loss:  0.5009759664535522 . Tr Accuracy:  58.26385483079941 . Ts Accuracy:  75.79505300353357 Fr:  0.14015365
# epoch:  3 . Loss:  0.8235954642295837 . Tr Accuracy:  80.44384502206964 . Ts Accuracy:  86.17491166077738 Fr:  0.12855339
# epoch:  4 . Loss:  0.1801595240831375 . Tr Accuracy:  85.93673369298676 . Ts Accuracy:  78.00353356890459 Fr:  0.12772396
# epoch:  5 . Loss:  0.6335716843605042 . Tr Accuracy:  87.84943599803826 . Ts Accuracy:  83.25971731448763 Fr:  0.122699216
# epoch:  6 . Loss:  0.4184652864933014 . Tr Accuracy:  88.46248160863168 . Ts Accuracy:  85.02650176678445 Fr:  0.12641016
# epoch:  7 . Loss:  0.11178665608167648 . Tr Accuracy:  91.66257969592938 . Ts Accuracy:  83.70141342756183 Fr:  0.11583985
# epoch:  8 . Loss:  0.24507765471935272 . Tr Accuracy:  93.80823933300637 . Ts Accuracy:  80.43286219081273 Fr:  0.11689453
# epoch:  9 . Loss:  0.02580219879746437 . Tr Accuracy:  93.6856302108877 . Ts Accuracy:  78.53356890459364 Fr:  0.11530859
# epoch:  10 . Loss:  0.132295161485672 . Tr Accuracy:  95.15693967631192 . Ts Accuracy:  87.0583038869258 Fr:  0.107161455
# epoch:  11 . Loss:  0.052720583975315094 . Tr Accuracy:  97.4619911721432 . Ts Accuracy:  87.54416961130742 Fr:  0.111710936
# epoch:  12 . Loss:  0.013912809081375599 . Tr Accuracy:  97.63364394310936 . Ts Accuracy:  88.29505300353357 Fr:  0.11046224
# epoch:  13 . Loss:  0.013542967848479748 . Tr Accuracy:  98.43060323688083 . Ts Accuracy:  84.31978798586573 Fr:  0.11176432
# epoch:  14 . Loss:  0.03696903958916664 . Tr Accuracy:  98.14860225600785 . Ts Accuracy:  85.95406360424029 Fr:  0.113291666
# epoch:  15 . Loss:  0.12882192432880402 . Tr Accuracy:  98.4428641490927 . Ts Accuracy:  84.67314487632508 Fr:  0.11321094
# epoch:  16 . Loss:  0.008622714318335056 . Tr Accuracy:  98.60225600784699 . Ts Accuracy:  87.72084805653711 Fr:  0.11448698
# epoch:  17 . Loss:  0.08007299154996872 . Tr Accuracy:  98.58999509563512 . Ts Accuracy:  85.07067137809187 Fr:  0.109397136
# epoch:  18 . Loss:  0.17344336211681366 . Tr Accuracy:  97.91564492398234 . Ts Accuracy:  86.43992932862191 Fr:  0.114264324
# epoch:  19 . Loss:  0.31412410736083984 . Tr Accuracy:  97.89112309955861 . Ts Accuracy:  90.68021201413427 Fr:  0.110308595
# Mean FR:  0.10995052
#  Accuracy:  90.7243816254417