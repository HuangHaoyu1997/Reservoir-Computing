import numpy as np
import matplotlib.pyplot as plt
import os, time, torch, ray, pickle
from scipy.linalg import pinv


from utils import encoding, A_initial, activation
import warnings

warnings.filterwarnings("ignore")
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
        
        self.r_history = np.zeros((self.N_hid))
        self.mem = np.zeros((self.N_hid))
        # self.spike = np.zeros((self.N_hid))
    
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
        
    def membrane(self, x, spike):
        mem = self.mem * self.decay * (1-spike) + x
        spike = np.array(mem>self.thr, dtype=np.float32)
        self.mem = mem
        return spike
    
    def softmax(self, x):
        return np.exp(x)/np.exp(x).sum()
    
    def forward(self, x):
        '''
        一个样本的长度应该超过1,即由多帧动态数据构成
        r.shape
        y.shape
        spike_train.shape (frame, N_hid)
        '''
        assert x.shape[0]>1
        spike_train = []
        spike = np.zeros((self.N_hid))
        timestep = x.shape[0]
        for t in range(timestep):
            Ar = np.matmul(self.A, self.r_history)
            U = np.matmul(self.W_in, x[t,:])
            r = (1 - self.alpha) * self.r_history + self.alpha * activation(Ar + U)
            spike = self.membrane(r, spike)
            spike_train.append(spike)
            self.r_history = r
            
        y = np.matmul(self.W_out, r)
        y = self.softmax(y)
        return r, y, np.array(spike_train)

if __name__ == '__main__':
    
    
    # ray.init()
    
    train_loader, test_loader = MNIST_generation(train_num=500,
                                                 test_num=250,
                                                 batch_size=1)
    model = RC(N_input=28*28,
               N_hidden=1000,
               N_output=10,
               alpha=0.8,
               decay=0.5,
               threshold=1.0,
               R=0.3,
               p=0.25,
               gamma=1.0,
               
               )
    
    learn(model, train_loader, frames=10)
    
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
