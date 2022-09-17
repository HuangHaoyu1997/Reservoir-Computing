import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
import os, time, torch, pickle
from scipy.linalg import pinv
from utils import encoding, A_initial, activation, softmax

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class RC:
    '''
    Reservoir Computing Model
    '''
    def __init__(self,
                 N_input, # 输入维度
                 N_hidden, # reservoir神经元数量
                 N_output, # 输出维度
                 alpha, # memory factor
                 decay, # membrane potential decay factor
                 threshold, # firing threshold
                 R, # distance factor
                 p, # 
                 gamma,
                 ) -> None:
        
        self.N_in = N_input
        self.N_hid = N_hidden
        self.N_out = N_output
        self.alpha = alpha
        self.decay = decay
        self.thr = threshold
        self.R = R
        self.p = p
        self.gamma = gamma
        self.random_init = True, # 初始状态是否随机初始化
        
        self.reset()
    
        
    def reset(self,):
        '''
        random initialization:
        W_in:      input weight matrix
        A:         reservoir weight matrix
        W_out:     readout weight matrix
        r_history: state of reservoir neurons
        mem:       membrane potential of reservoir neurons
        
        '''
        self.W_in = np.random.uniform(low=np.zeros((self.N_hid, self.N_in)), 
                                      high=np.ones((self.N_hid, self.N_in))*0.1)
        
        self.A = A_initial(self.N_hid, self.R, self.p, self.gamma)
        # self.A = np.random.uniform(low=-1*np.ones((self.N_hid, self.N_hid)), 
        #                            high=np.ones((self.N_hid, self.N_hid)))
        
        # 用系数0.0533缩放，以保证谱半径ρ(A)=1.0
        self.W_out = np.random.uniform(low=-0.0533*np.ones((self.N_out, self.N_hid)), 
                                       high=0.0533*np.ones((self.N_out, self.N_hid)))
        
    
    def state_dict(self,):
        return {
            'W_in': self.W_in,
            'A': self.A,
            'W_out': self.W_out,
            'N_input': self.N_in,
            'N_hidden': self.N_hid,
            'N_output': self.N_out,
            'alpha': self.alpha,
            'decay': self.decay,
            'threshold': self.thr,
        }
        
    def membrane(self, mem, x, spike):
        '''
        update membrane voltage for reservoir neurons
        
        mem   [batch, N_hid]
        x     [batch, N_hid]
        spike [batch, N_hid]
        '''
        # print(mem.shape, spike.shape, x.shape)
        mem = mem * self.decay * (1-spike) + x
        spike = np.array(mem>self.thr, dtype=np.float32)
        return mem, spike
    
    def forward_(self, x):
        '''
        inference function of spiking version
        
        x: input vector
        
        return 
        
        mems: [frames, batch, N_hid]
        
        spike_train: [frames, batch, N_hid]
        
        '''
        batch = x.shape[0]
        frames = x.shape[1]
        spike_train = []
        spike = np.zeros((batch, self.N_hid), dtype=np.float32)
        mem = np.random.uniform(0, 0.2, size=(batch, self.N_hid))
        # mem = np.zeros((batch, self.N_hid), dtype=np.float32)
        mems = []
        for t in range(frames):
            # x[:,t,:].shape (batch, N_in)
            U = np.matmul(x[:,t,:], self.W_in.T) # (batch, N_hid)
            r = np.matmul(spike, self.A) # (batch, N_hid)
            y = self.alpha * r + (1-self.alpha) * U
            y = activation(y)
            mem, spike = self.membrane(mem, y, spike)
            spike_train.append(spike)
            mems.append(mem)
        
        return np.array(mems), np.array(spike_train)
    
    def forward(self, x):
        '''
        一个样本的长度应该超过1,即由多帧动态数据构成
        
        r.shape (batch, N_hid)
        
        y.shape (batch, N_out)
        
        spike_train.shape (frame, batch, N_hid)
        
        '''
        assert x.shape[1]>1
        
        batch = x.shape[0]
        frames = x.shape[1]
        
        spike_train = []
        spike = np.zeros((batch, self.N_hid), dtype=np.float32)
        r_history = np.zeros((batch, self.N_hid), dtype=np.float32)
        mem = np.zeros((batch, self.N_hid), dtype=np.float32)
        
        for t in range(frames):
            
            Ar = np.matmul(r_history, self.A) # (batch, N_hid)
            
            U = np.matmul(x[:,t,:], self.W_in.T) # (batch, N_hid)
            r = (1 - self.alpha) * r_history + self.alpha * activation(Ar + U)
            mem, spike = self.membrane(mem, r, spike)
            spike_train.append(spike)
            r_history = r
            
        y = np.matmul(r, self.W_out.T)
        # y = softmax(y)
        return r, y, np.array(spike_train)

if __name__ == '__main__':
    
    from data import MNIST_generation
    # ray.init()
    
    train_loader, test_loader = MNIST_generation(train_num=32,
                                                 test_num=250,
                                                 batch_size=16)
    
    model = RC(N_input=28*28,
               N_hidden=1000,
               N_output=10,
               alpha=0.8,
               decay=0.5,
               threshold=1.3,
               R=0.2,
               p=0.25,
               gamma=1.0,
               
               )
    for i, (images, lables) in enumerate(train_loader):
        enc_img = encoding(images, frames=20)
        mems, spike_train = model.forward_(enc_img)
        # r, y, spike_train = model.forward(enc_img)
        firing_rate = spike_train.sum(0)/20
    
    
    plt.figure()
    for i in range(4):
        for j in range(4):
            plt.subplot(4,4,4*i+j+1)
            plt.hist(firing_rate[4*i+j,:])
    plt.show()
    
    # learn(model, train_loader, frames=10)
    
    # from cma import CMAEvolutionStrategy
    # es = CMAEvolutionStrategy(x0=np.zeros((model.N_hid*model.N_out)),
    #                             sigma0=0.5,
    #                             #   inopts={
    #                             #             'popsize':100,
    #                             #           },
    #                             )
    # N_gen = 100
    # for g in range(N_gen):
    #     solutions = es.ask()
    #     task_list = [train.remote(model,
    #                                 solution,
    #                                 train_loader, 
    #                                 test_loader, 
    #                                 batch_size=1,
    #                                 frames=10,
    #                                 ) for solution in solutions]
    #     fitness = ray.get(task_list)
    #     es.tell(solutions, fitness)
    #     with open('ckpt_'+str(g)+'.pkl', 'wb') as f:
    #         pickle.dump([solutions, fitness], f)
    #     print(np.min(fitness))
    
    # labels = []
    # for i, (image, label) in enumerate(train_loader):
    #     label_ = torch.zeros(1, 10).scatter_(1, label.view(-1, 1), 1).squeeze().numpy()
    #     labels.append(label_)
    # labels = np.array(labels, dtype=np.float).T
    
    # with open('train_labels.pkl', 'rb') as f:
    #     labels = pickle.load( f)
        
    # with open('rs.pkl', 'rb') as f:
    #     R_T = pickle.load(f)
    
    # R = R_T.T
    # R_inv = pinv(R)
    # W_out = np.matmul(labels, R_inv)
    # print(W_out.shape)
    # model.W_out = W_out
    
    # correct = 0
    # for i, (image, label) in enumerate(train_loader):
    #     image = encoding(image.squeeze(), 10) # shape=(30,784)
    #     r, y, _ = model.forward(image)
    #     predict = y.argmin()
    #     correct += float(predict == label.item())
    # print(correct / len(train_loader))
    
    # rs = inference.remote(model, train_loader, 10)
    # rs = ray.get(rs)
    # 
    # print(rs.shape)
    # with open('rs.pkl', 'wb') as f:
    #     pickle.dump(rs, f)
    
    # y, spike_train = model.forward(data)
    # spike_train = np.array(spike_train)
    # print(y, spike_train.shape)
    # plt.imshow(spike_train[:,0:100])
    # plt.pause(10)
    # ray.shutdown() 
