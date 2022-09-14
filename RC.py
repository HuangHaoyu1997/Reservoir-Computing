import numpy as np
import matplotlib.pyplot as plt
import os, time, torch, ray, pickle, random, torchvision
from scipy.linalg import pinv
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
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

def readout_sk(X_train, 
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



# @ray.remote
def inference(model:RC,
              train_loader,
              frames
              ):
    '''
    给定数据集和模型, 推断reservoir state vector
    '''
    rs = []
    start_time = time.time()
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
        
    print('Time elasped:', time.time() - start_time)
    return np.array(rs), np.array(spikes), np.array(labels)

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
    tr_score, val_score, = readout_sk(train_rs.T, 
                                    #   val_rs.T, 
                                      test_rs.T, 
                                      train_label, 
                                    #   val_label, 
                                      test_label)
    print(tr_score, val_score)
    return val_score


'''
correct = 0
total = 0
for batch_idx, (inputs, targets) in enumerate(test_loader):
    inputs = encoding(inputs.squeeze(), frames=frames) # shape=(30,784)
    outputs, _ = model.forward(inputs)
    labels_ = torch.zeros(batch_size, 10).scatter_(1, targets.view(-1, 1), 1).squeeze().numpy()
    loss = cross_entropy(labels_, outputs)
    predicted = outputs.argmax()
    total += float(targets.size(0))
    correct += float(predicted==targets.item())
    if batch_idx %100 ==0:
        acc = 100. * float(correct) / float(total)
        print(batch_idx, len(test_loader),' Acc: %.5f' % acc)

print('Iters:', epoch,'\n\n\n')
print('Test Accuracy of the model on the 10000 test images: %.3f' % (100 * correct / total))
acc = 100. * float(correct) / float(total)
acc_record.append(acc)
if epoch % 5 == 0:
    print(acc)
    print('Saving..')
    state = {
        'net': model.state_dict(),
        'acc': acc,
        'epoch': epoch,
        'acc_record': acc_record,
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, './checkpoint/ckpt' + names + '.t7')
    best_acc = acc
'''


def MNIST_generation(train_num=1000, test_num=250, batch_size=1):
    '''
    生成随机编码的MNIST动态数据集
    train_num: 训练集样本数
    '''
    
    train_dataset = torchvision.datasets.MNIST(root='./reservoir/data/', 
                                               train=True, 
                                               download=False, 
                                               transform=transforms.ToTensor())
    
    # 只取一部分数据
    assert train_num<= len(train_dataset)
    idx = random.sample(list(range(len(train_dataset))), train_num)
    train_dataset.data = train_dataset.data[idx]
    
    train_loader = torch.utils.data.DataLoader(train_dataset, 
                                               batch_size=batch_size, 
                                               shuffle=True, 
                                               num_workers=0)

    test_dataset = torchvision.datasets.MNIST(root='./reservoir/data/', 
                                          train=False, 
                                          download=False, 
                                          transform=transforms.ToTensor())
    
    # 只取一部分数据
    assert test_num<= len(test_dataset)
    idx = random.sample(list(range(len(test_dataset))), test_num)
    test_dataset.data = test_dataset.data[idx]
    
    test_loader = torch.utils.data.DataLoader(test_dataset, 
                                              batch_size=batch_size, 
                                              shuffle=False, 
                                              num_workers=0)
    return train_loader, test_loader



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
