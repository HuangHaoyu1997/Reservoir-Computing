'''
train the pytorch version Reservoir Computing model in GPUs
'''

import time
import torch
import pickle
import numpy as np
from RC import MLP, torchRC
from utils import encoding
from data import MNIST_generation
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from openbox import Optimizer, sp, ParallelOptimizer

def inference(model:torchRC,
              data_loader,
              ):
    '''
    给定数据集和模型, 推断reservoir state vector
    '''
    device = model.device
    # start_time = time.time()
    labels = []
    spikes = None
    
    for i, (image, label) in enumerate(data_loader):
        
        print('batch', i)
        mems, spike = model(image.to(device)) # [batch, frames, N_hid], [batch, frames, N_hid]
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
                      epochs,
                      X_train,
                      X_test,
                      y_train,
                      y_test,):
    
    cost = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    for epoch in range(epochs) :
        sum_loss = 0
        train_correct = 0
        for i in range(30):
            x = X_train[i*2000:(i+1)*2000]
            y = y_train[i*2000:(i+1)*2000]
            out = model(x)

            optimizer.zero_grad()
            loss = cost(out, y)
            loss.backward()
            optimizer.step()
    
            _, id = torch.max(out.data, 1)
            sum_loss += loss.data
            train_correct+=torch.sum(id==y.data)
        print('[%d,%d] loss:%.03f' % (epoch + 1, epochs, sum_loss / 30))
        print('        correct:%.03f%%' % (100 * train_correct / 60000))
        print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
    model.eval()
    test_correct = 0
    for i in range(5):
        x = X_test[i*2000:(i+1)*2000]
        y = y_test[i*2000:(i+1)*2000]
        outputs = model(x)
        _, id = torch.max(outputs.data, 1)
        test_correct += torch.sum(id == y.data)
    print("correct:%.3f%%" % (100 * test_correct / 10000))
    
    return train_correct / 60000, test_correct / 10000


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

def learn(model:torchRC, train_loader, test_loader):
    # rs.shape (500, 1000)
    # labels.shape (500,)
    train_rs, train_label = inference(model, train_loader,)
    
    test_rs, test_label = inference(model, test_loader,)
    
    mlp = MLP(2000, 128, 10).to(model.device)
    tr_score, te_score, = train_mlp_readout(model=mlp, 
                                            epochs=20,
                                            X_train=train_rs,
                                            X_test=test_rs,
                                            y_train=train_label,
                                            y_test=test_label)
    
    # tr_score, te_score, = learn_readout(train_rs.T, 
    #                                      test_rs.T, 
    #                                      train_label, 
    #                                      test_label)
    print(tr_score, te_score)
    return -te_score # openbox 默认最小化loss

def config_model(config):
    model = torchRC(N_input=28*28,
                    N_hidden=1000,
                    N_output=10,
                    alpha=config['alpha'],
                    decay=config['decay'],
                    threshold=config['thr'],
                    R=config['R'],
                    p=config['p'],
                    gamma=config['gamma'],
                    sub_thr=False,
                    binary=False,
                    frames=config['frames'],
                    device=config['device'],
                    )
    return model

def rollout(config):
    model = config_model(config)
    train_loader, test_loader = MNIST_generation(batch_size=2000) # batch=2000 速度最快
    loss = learn(model, train_loader, test_loader)
    return {'objs': (loss,)}


def param_search(run_time):
    # Define Search Space
    space = sp.Space()
    x1 = sp.Real(name="alpha", lower=0, upper=1, default_value=0.5)
    x2 = sp.Real(name="decay", lower=0, upper=2, default_value=0.5)
    x3 = sp.Real(name="thr", lower=0, upper=2, default_value=0.7) 
    x4 = sp.Real(name="R", lower=0.05, upper=0.5, default_value=0.3) 
    x5 = sp.Real(name="p", lower=0, upper=1, default_value=0.5) 
    x6 = sp.Real(name="gamma", lower=0, upper=2, default_value=1.0) 
    space.add_variables([x1, x2, x3, x4, x5, x6])
    
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
    
    # param_search(run_time)
    inference()