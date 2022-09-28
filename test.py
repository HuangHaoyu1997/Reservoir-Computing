from train import *
import matplotlib.pyplot as plt
from utils import *

if __name__ == '__main__':
    config = {
        'alpha':0.8,
        'decay':0,
        'thr': 0.3,
        'R': 0.2,
        'p': 0.25,
        'gamma': 1.0,
    }

    model = config_model(config)
    # print(1000**2-(model.A==0).sum())
    # print(spectral_radius(model.A))
    
    # plt.imshow(model.A)
    # plt.show()
    # train_loader, test_loader = MNIST_generation(train_num=100,
    #                                              test_num=250,
    #                                              batch_size=13)
    
    # inference(model, train_loader, frames=20)
    
    t = time.time()
    rollout(config)
    print(time.time()-t)

