import cma
import torch
from main_sparch_better import *

model = torch.load(r'log\08-22-02-36\ckpt\best_model_87_93.7279151943463.pth').to('cuda')
train_loader, test_loader = load_shd_or_ssc()

def objective_func(x):
    alpha = x[0: 128]
    beta = x[128: 128*2]
    a = x[128*2: 128*3]
    b = x[128*3: 128*4]
    model.eval()
    with torch.no_grad():
        for i in range(128):
            model.alpha.data[i*8:(i+1)*8] = torch.tensor(alpha[i]).cuda() # *(np.exp(-1 / 25) - np.exp(-1 / 5)) + np.exp(-1 / 5)
            model.beta.data[i*8:(i+1)*8] = torch.tensor(beta[i]).cuda()
            model.a.data[i*8:(i+1)*8] = torch.tensor(a[i]).cuda()
            model.b.data[i*8:(i+1)*8] = torch.tensor(b[i]).cuda()
        losses, correct, total = [], 0, 0
        for images, labels in test_loader:
            images = torch.sign(images.clamp(min=0)) # all pixels should be 0 or 1
            outputs, firing_rates, all_spikes = model(images.to(config.device), 0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted.cpu() == labels.long().cpu()).sum()
        accuracy = 100. * correct.numpy() / total

    return 93.7279151943463-accuracy

# 定义初始参数和每个参数维度上的步长
# x0 = np.random.uniform(np.exp(-1/5), np.exp(-1/25), size=(128))  # 初始参数值
x0 = np.concatenate((np.random.uniform(np.exp(-1/5), np.exp(-1/25), 128), 
                np.random.uniform(np.exp(-1/30), np.exp(-1/120), 128),
                np.random.uniform(-1, 1, 128),
                np.random.uniform(0, 2, 128),))
sigma0 = 0.01  # 参数步长

# 定义参数的上下界
lower_bounds = [np.exp(-1/5)]*128 + [np.exp(-1 / 30)]*128 + [-1]*128 + [0]*128  # 参数的下界
upper_bounds = [np.exp(-1/25)]*128 + [np.exp(-1 / 120)]*128 + [1]*128 + [2]*128  # 参数的上界
bounds = [lower_bounds, upper_bounds]

# 创建CMAES对象并运行优化
es = cma.CMAEvolutionStrategy(x0, sigma0, {'bounds': bounds,
                                           'popsize': 50,
                                           'maxiter': 1000,
                                          #  'verbose': True,
                                           'verb_disp': 1,
                                           'tolfun': 1e-11, 
                                           'tolstagnation': 50})
es.optimize(objective_func)

# 输出最佳解和对应的目标函数值
best_solution = es.result.xbest
best_fitness = es.result.fbest
print("Best solution found: ", best_solution)
print("Best fitness value: ", best_fitness)