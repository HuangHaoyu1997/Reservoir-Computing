import time
import pickle
import numpy as np
from RC import RC
from utils import encoding
from data import MNIST_generation
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from openbox import Optimizer, sp, ParallelOptimizer

# @ray.remote
def inference(model:RC,
              train_loader,
              frames
              ):
    '''
    给定数据集和模型, 推断reservoir state vector
    '''
    rs = []
    # start_time = time.time()
    labels = []
    spikes = []
    for i, (image, label) in enumerate(train_loader):
        # static img -> random firing sequence
        image = encoding(image.squeeze(), frames) # shape=(30,784)
        
        # spike.shape (frame, N_hid)=(5, 5000)
        r, y, spike = model.forward(image)
        spike_sum = spike.sum(0)
        
        # label_ = torch.zeros(batch_size, 10).scatter_(1, label.view(-1, 1), 1).squeeze().numpy()
        # loss = cross_entropy(label_, outputs)
        
        rs.append(r)
        spikes.append(spike_sum)
        labels.append(label.item())
        
    # print('Time elasped:', time.time() - start_time)
    return np.array(rs), np.array(spikes), np.array(labels)

def learn_readout(X_train, 
               X_validation, 
               # X_test, 
               y_train, 
               y_validation, 
               # y_test,
               ):
    '''
    X_train: shape(r_dim, num)
    y_train: shape(num, )
    
    accuracy_score返回分类精度,最高=1
    '''
    lr = LogisticRegression(solver='lbfgs',
                            multi_class='multinomial',
                            verbose=False,
                            max_iter=200,
                            n_jobs=-1,
                            )
    lr.fit(X_train.T, y_train.T)
    y_train_predictions = lr.predict(X_train.T)
    y_validation_predictions = lr.predict(X_validation.T)
    # y_test_predictions = lr.predict(X_test.T)
    
    return accuracy_score(y_train_predictions, y_train.T), \
            accuracy_score(y_validation_predictions, y_validation.T), \
            # accuracy_score(y_test_predictions, y_test.T)

def learn(model, train_loader, frames):
    # rs.shape (500, 1000)
    # labels.shape (500,)
    rs, spikes, labels = inference(model,
                            train_loader,
                            frames,
                            )
    # print(spikes.shape, labels.shape)
    train_rs = spikes[:300]
    train_label = labels[:300]
    test_rs = spikes[300:]
    test_label = labels[300:]
    # val_rs = spikes[400:]
    # val_label = labels[400:]
    tr_score, val_score, = learn_readout(train_rs.T, 
                                         # val_rs.T, 
                                         test_rs.T, 
                                         train_label, 
                                         # val_label, 
                                         test_label)
    print(tr_score, val_score)
    return -val_score # openbox 默认最小化loss

def config_model(config):
    model = RC(N_input=28*28,
                N_hidden=1000,
                N_output=10,
                alpha=config['alpha'],
                decay=config['decay'],
                threshold=config['thr'],
                R=config['R'],
                p=config['p'],
                gamma=config['gamma'],
                )
    return model

def rollout(config):
    model = config_model(config)
    train_loader, test_loader = MNIST_generation(train_num=500,
                                                 test_num=250,
                                                 batch_size=1)
    loss = learn(model, train_loader, frames=10)
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
    
    param_search(run_time)
    