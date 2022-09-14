import time
import numpy as np
from RC import RC
from utils import encoding
from data import MNIST_generation
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

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
    return val_score

if __name__ == '__main__':
    
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