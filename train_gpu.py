'''
train the pytorch version Reservoir Computing model in GPUs
'''

from distutils.command.config import config
import time
from turtle import done
import torch
import random
import pickle
import numpy as np
from RC import MLP, torchRC
from config import Config
from utils import encoding
from data import PoissonDataset, part_DATA
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from openbox import Optimizer, sp, ParallelOptimizer

def inference(model:torchRC, config:Config, data_loader,):
    '''
    给定数据集和模型, 推断reservoir state vector
    '''
    
    device = model.device
    frames = model.frames
    device = model.device
    N_in = model.N_in
    # start_time = time.time()
    labels = []
    spikes = None
    
    for i, (image, label) in enumerate(data_loader):
        print('batch', i)
        batch = image.shape[0]
        if config.data == 'poisson':
            x_enc = image
        else:
            x_enc = None
            for _ in range(frames):
                spike = (image > torch.rand(image.size())).float()
                if x_enc is None: x_enc = spike
                else: x_enc = torch.cat((x_enc, spike), dim=1)
            x_enc = x_enc.view(batch, frames, N_in) # [batch, frames, N_in]
        
        mems, spike = model(x_enc.to(device)) # [batch, frames, N_hid], [batch, frames, N_hid]
        # concat membrane and spike train as representation
        concat = torch.cat((mems, spike), dim=-1) # [batch, frames, 2*N_hid]
        concat = concat.mean(1) # [batch, 2*N_hid]
        if spikes is None: spikes = concat # spikes = spike_sum
        else: spikes = torch.cat((spikes, concat), dim=0)
        labels.extend(label.numpy().tolist())
    # print('Time elasped:', time.time() - start_time)
    
    # return spikes.detach().cpu().numpy(), np.array(labels)
    return spikes.detach(), torch.tensor(labels).to(device)

def train_mlp_readout(model:MLP,
                      config:Config,
                      X_train,
                      X_test,
                      y_train,
                      y_test,
                      ):
    train_num = X_train.shape[0]
    test_num = X_test.shape[0]
    iteration = int(train_num/config.batch_size)
    iter = int(test_num/config.batch_size)
    cost = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    for epoch in range(config.epoch) :
        model.train()
        sum_loss = 0
        train_correct = 0
        for i in range(iteration):
            x = X_train[i*config.batch_size:(i+1)*config.batch_size]
            y = y_train[i*config.batch_size:(i+1)*config.batch_size]
            out = model(x)

            optimizer.zero_grad()
            loss = cost(out, y)
            loss.backward()
            optimizer.step()
    
            _, id = torch.max(out.data, 1)
            sum_loss += loss.data
            train_correct+=torch.sum(id==y.data)
        
        model.eval()
        test_correct = 0
        for i in range(iter):
            x = X_test[i*config.batch_size:(i+1)*config.batch_size]
            y = y_test[i*config.batch_size:(i+1)*config.batch_size]
            outputs = model(x)
            _, id = torch.max(outputs.data, 1)
            test_correct += torch.sum(id == y.data)
        if config.verbose:
            print('[%d,%d] loss:%.03f, train acc:%.4f, test acc:%.4f' % (epoch+1, config.epoch, sum_loss/iteration, train_correct/train_num, test_correct/test_num))
        
    return train_correct / train_num, test_correct / test_num

class AttackGym:
    def __init__(self, config:Config):
        self.config = config
        
        self.reset()
        
    def reset(self,):
        config = self.config
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        self.model = torchRC(config)
        self.T = 0
    
    def step(self, action):
        # reservoir attacking
        self.model.W_ins[0] = None
        
        obs, reward = rollout_attack(self.config, self.model)
        
        if self.T > self.config.episode_len:
            done = True
        else:
            done = False
        return obs, reward, done

def attack(model:torchRC,
           config,
           
           ):
    # attack
    model.W_ins[0]
    
    # rollout
    reward = rollout_attack(config, model)

def rollout_attack(config:Config, model:torchRC):
    if config.data == 'poisson':
        train_loader, test_loader = PoissonDataset(config)
    else:
        train_loader, test_loader = part_DATA(config)
    loss = learn(model, train_loader, test_loader, config)
    return loss

def learn_readout(X_train, 
                  X_validation, 
                  y_train, 
                  y_validation, 
                  ):
    '''
    X_train: shape(r_dim, num)
    y_train: shape(num, )
    
    accuracy_score返回分类精度,最高=1
    '''
    lr = LogisticRegression(solver='lbfgs',
                            multi_class='auto', # multinomial
                            verbose=False,
                            max_iter=200,
                            n_jobs=-1,
                            )
    lr.fit(X_train.T, y_train.T)
    y_train_predictions = lr.predict(X_train.T)
    y_validation_predictions = lr.predict(X_validation.T)
    print(y_validation, y_validation_predictions)
    # y_test_predictions = lr.predict(X_test.T)
    
    return accuracy_score(y_train_predictions, y_train.T), \
            accuracy_score(y_validation_predictions, y_validation.T), \
            # accuracy_score(y_test_predictions, y_test.T)

def learn(model:torchRC, train_loader, test_loader, config:Config):
    train_rs, train_label = inference(model, config, train_loader,)
    test_rs, test_label = inference(model, config, test_loader,)
    
    mlp = MLP(2*config.N_hid, config.mlp_hid, config.N_out).to(model.device)
    train_score, test_score, = train_mlp_readout(model=mlp, 
                                                config=config,
                                                X_train=train_rs,
                                                X_test=test_rs,
                                                y_train=train_label,
                                                y_test=test_label)
    
    # tr_score, te_score, = learn_readout(train_rs.T, 
    #                                      test_rs.T, 
    #                                      train_label, 
    #                                      test_label)
    if config.verbose:
        print(train_score, test_score)
    return -test_score.detach().cpu().item() # openbox 默认最小化loss

def rollout(configuration):
    '''
    for bayesian optimization
    '''
    from config import Config
    config = Config
    
    config.alpha = configuration['alpha']
    config.p_in = configuration['p_in']
    config.gamma = configuration['gamma']
    config.binary = configuration['binary']
    config.noise_str = configuration['noise']
    config.m_BA = configuration['m_BA']
    config.k = configuration['k']
    config.LIF_decay = configuration['decay']
    config.LIF_thr = configuration['thr']
    
    model = torchRC(config)

    # train_loader, test_loader = MNIST_generation(batch_size=config.batch_size) # batch=2000 速度最快
    # loss = learn(model, train_loader, test_loader, config)
    
    train_loader, test_loader = part_DATA(config)
    loss = learn(model, train_loader, test_loader, config)
    print(loss)
    return {'objs': (loss,)}

def rollouts(config:Config):
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    model = torchRC(config)

    # train_loader, test_loader = MNIST_generation(batch_size=config.batch_size) # batch=2000 速度最快
    # loss = learn(model, train_loader, test_loader, config)
    if config.data == 'poisson':
        train_loader, test_loader = PoissonDataset(config)
    else:
        train_loader, test_loader = part_DATA(config)
    
    loss = learn(model, train_loader, test_loader, config)
    return {'objs': (loss,)}


def param_search(run_time):
    '''
    openbox 默认最小化loss
    '''
    
    # Define Search Space
    space = sp.Space()
    x1 = sp.Real(name="alpha", lower=0, upper=1, default_value=0.2)
    x2 = sp.Real(name="p_in", lower=0, upper=1, default_value=0.2) 
    x3 = sp.Real(name="gamma", lower=0, upper=2, default_value=1.0) 
    x4 = sp.Categorical(name='binary', choices=[0, 1], default_value=0)
    x5 = sp.Real(name='noise', lower=0, upper=1, default_value=0.05)
    x6 = sp.Int(name='m_BA', lower=1, upper=10, default_value=2)
    x7 = sp.Int(name='k', lower=2, upper=10, default_value=3)
    x8 = sp.Real(name="decay", lower=0, upper=2, default_value=0.5)
    x9 = sp.Real(name="thr", lower=0, upper=2, default_value=0.7)  
    space.add_variables([x1, x2, x3, x4, x5, x6, x7, x8, x9])
    
    # Parallel Evaluation on Local Machine 本机并行优化
    opt = ParallelOptimizer(rollout,
                            space,                      # 搜索空间
                            parallel_strategy='async',  # 'sync'设置并行验证是异步还是同步, 使用'async'异步并行方式能更充分利用资源,减少空闲
                            batch_size=4,               # 设置并行worker的数量
                            batch_strategy='default',   # 设置如何同时提出多个建议的策略, 推荐使用默认参数 ‘default’ 来获取稳定的性能。
                            num_objs=1,
                            num_constraints=0,
                            max_runs=3000,
                            # surrogate_type='gp',
                            surrogate_type='auto',
                            time_limit_per_trial=180,
                            task_id='parallel_async',
                            logging_dir='openbox_logs', # 实验记录的保存路径, log文件用task_id命名
                            random_state=123,
                            )
    history = opt.run()
    print(history)
    # print(history.get_importance()) # 输出参数重要性
    with open(run_time+'.pkl', 'wb') as f:
        pickle.dump(history, f)

if __name__ == '__main__':

    
    run_time = time.strftime("%Y.%m.%d-%H-%M-%S", time.localtime())
    
    param_search(run_time)
    # inference()