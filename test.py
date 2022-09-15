import numpy as np
import matplotlib.pyplot as plt
from openbox import Optimizer, sp, ParallelOptimizer
import warnings
warnings.filterwarnings("ignore")

# Define Search Space
space = sp.Space()
x1 = sp.Real(name="x1", lower=-5, upper=10, default_value=0)
x2 = sp.Real(name="x2", lower=0, upper=15, default_value=0)
x3 = sp.Int(name="x3", lower=0, upper=100) 
x4 = sp.Categorical(name="x4", 
                    choices=["rbf", "poly", "sigmoid"], 
                    default_value="rbf")
space.add_variables([x1, x2])

# Define Objective Function
# OpenBox默认执行最小化
def branin(config):
    x1, x2 = config['x1'], config['x2']
    # y = (x2-5.1/(4*np.pi**2)*x1**2 + 5/np.pi*x1-6)**2 + 10*(1-1/(8*np.pi))*np.cos(x1)+10
    y = (x2-5.1*x1**2 + 5*x1-6)**2 -10
    # return y
    return {'objs': (y,)}


# Run
if __name__ == '__main__':


    # Parallel Evaluation on Local Machine 本机并行优化
    opt = ParallelOptimizer(branin,
                            space,                      # 搜索空间
                            parallel_strategy='async',  # 'sync'设置并行验证是异步还是同步, 使用'async'异步并行方式能更充分利用资源,减少空闲
                            batch_size=4,               # 设置并行worker的数量
                            batch_strategy='default',   # 设置如何同时提出多个建议的策略, 推荐使用默认参数 ‘default’ 来获取稳定的性能。
                            num_objs=1,
                            num_constraints=0,
                            max_runs=50,
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
    
    history.plot_convergence(xlabel="Number of iterations $n$",
                             ylabel=r"Min objective value after $n$ iterations",
                             true_minimum=0.397887,
                             )
    plt.show()