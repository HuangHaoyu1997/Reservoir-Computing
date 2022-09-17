from train import *

if __name__ == '__main__':
    config = {
        'alpha':0.5,
        'decay':0.5,
        'thr': 1.2,
        'R': 0.3,
        'p': 0.25,
        'gamma': 1.0,
    }
    model = config_model(config)
    
    train_loader, test_loader = MNIST_generation(train_num=100,
                                                 test_num=250,
                                                 batch_size=13)
    
    inference(model, train_loader, frames=20)

