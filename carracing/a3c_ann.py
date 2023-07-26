import torch, os, gym, time, glob, sys, warnings
from collections import deque
warnings.filterwarnings("ignore")
import numpy as np
from scipy.signal import lfilter
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.distributions import Normal
os.environ['OMP_NUM_THREADS'] = '1'

class config:
    # RL parameter
    env = 'CarRacing-v2'
    date = time.strftime("%Y-%m-%d-%H-%M-%S/", time.localtime(time.time()))[5:16]
    save_dir = './log/' + date
    processes = 8
    render = False
    test = False
    horizon = 0.0 # moving avg, 越接近1越平滑
    seed = 12345
    gamma = 0.99 # discount factor
    tau = 1.0
    max_frame = 5e6
    max_frame_episode = 5e3
    ckpt_frame = 1e6
    pomdp = False
    
    # Training parameter
    input_learn = False
    update_freq = 20
    lr = 1e-3
    entropy_loss = 0.01
    
    # SNN parameter
    input = gym.make(env).observation_space.shape[0]-2
    hid = 256
    output = gym.make(env).action_space.shape[0]
    gradient_type = 'MG' # 'G', 'slayer', 'linear' 窗型函数
    scale = 6.        # special for 'MG'
    hight = 0.15      # special for 'MG'
    lens = 0.5  # hyper-parameters of approximate function
    surogate_gamma = 0.5
    b_j0 = 0.01
    R_m = 1.0
    time_step = 8 # inference steps
    dropout = 0.8
    dropout_stepping = 0.03
    dropout_stop = 0.97
    dropout_feq = 0.1e6
    device = 'cpu'

discount = lambda x, gamma: lfilter([1], [1,-gamma], x[::-1])[::-1] # discounted rewards one liner

def printlog(s, end='\n', mode='a'):
    print(s, end=end) ; f=open(config.save_dir+'/log.txt',mode) ; f.write(s+'\n') ; f.close()

class NNPolicy(nn.Module):
    def __init__(self):
        super(NNPolicy, self).__init__()
        
        self.conv1 = nn.Conv2d(8, 16, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 16, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(16, 16, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(16, 32, 3, stride=2, padding=1)
        
        self.inpt_hid1 = nn.Linear(1152, config.hid)
        self.hid1_critic = nn.Linear(config.hid, 1)
        self.hid1_actor = nn.Linear(config.hid, config.output)
        self.log_std = nn.Parameter(torch.ones(1, config.output))
        
    def forward(self, input):
        x = F.relu(self.conv1(input.float()))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(1, -1)
        x = F.relu(self.inpt_hid1(x))
        action = F.relu(self.hid1_actor(x))
        value = F.relu(self.hid1_critic(x))
        std   = self.log_std.exp().expand_as(action)
        dist  = Normal(action, std)
        return value, dist
    
    def try_load(self, save_dir):
        paths = glob.glob(save_dir + '*.tar') ; step = 0
        if len(paths) > 0:
            ckpts = [int(s.split('.')[-2]) for s in paths]
            ix = np.argmax(ckpts) ; step = ckpts[ix]
            self.load_state_dict(torch.load(paths[ix]))
        print("\tno saved models") if step == 0 else print("\tloaded model: {}".format(paths[ix]))
        return step
    
        
class SharedAdam(torch.optim.Adam): # extend a pytorch optimizer so it shares grads across processes
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        super(SharedAdam, self).__init__(params, lr, betas, eps, weight_decay)
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['shared_steps'], state['step'] = torch.zeros(1).share_memory_(), 0
                state['exp_avg'] = p.data.new().resize_as_(p.data).zero_().share_memory_()
                state['exp_avg_sq'] = p.data.new().resize_as_(p.data).zero_().share_memory_()
                
        def step(self, closure=None):
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None: continue
                    self.state[p]['shared_steps'] += 1
                    self.state[p]['step'] = self.state[p]['shared_steps'][0] - 1 # a "step += 1"  comes later
            super.step(closure)

def cost_func(values, logps, actions, rewards):
    np_values = values.view(-1).data.numpy()

    # generalized advantage estimation using \delta_t residuals (a policy gradient method)
    delta_t = np.asarray(rewards) + config.gamma * np_values[1:] - np_values[:-1]
    # logpys = logps.gather(1, actions.view(-1,1))
    gen_adv_est = discount(delta_t, config.gamma * config.tau)
    # policy_loss = -(logpys.view(-1) * torch.FloatTensor(gen_adv_est.copy())).sum()
    policy_loss = -(logps.sum(-1) * torch.FloatTensor(gen_adv_est.copy())).sum()
    
    # l2 loss over value estimator
    rewards[-1] += config.gamma * np_values[-1]
    discounted_r = discount(np.asarray(rewards), config.gamma)
    discounted_r = torch.tensor(discounted_r.copy(), dtype=torch.float32)
    value_loss = .5 * (discounted_r - values[:-1,0]).pow(2).sum()

    entropy_loss = (-logps * torch.exp(logps)).sum() # entropy definition, for entropy regularization
    return policy_loss + 0.5 * value_loss - config.entropy_loss * entropy_loss

def train(shared_model, shared_optimizer, rank, info, masks):
    # print(rank,'begin')
    env = gym.make(config.env) # make a local (unshared) environment
    env.reset(seed=config.seed + rank) ; torch.manual_seed(config.seed + rank); np.random.seed(config.seed + rank) # seed everything
    model = NNPolicy() # a local/unshared model
    state_deque = deque(maxlen=config.time_step)
    tmp = env.reset()[0]
    tmp = tmp.mean(-1)/255
    if config.pomdp: tmp += np.random.random(tmp.shape) * 0.3
    for _ in range(config.time_step):
        state_deque.append(tmp)
    
    # state = torch.tensor(env.reset()[0]) # get first state

    start_time = last_disp_time = time.time()
    episode_length, epr, eploss, done  = 0, 0, 0, True # bookkeeping

    while info['frames'][0] <= config.max_frame or config.test:
        model.load_state_dict(shared_model.state_dict()) # sync with main shared model

        values, logps, actions, rewards = [], [], [], [] # save values for computing gradients

        for step in range(config.update_freq):
            episode_length += 1
            value, dist = model(torch.tensor(state_deque).view(1, config.time_step, 96, 96))
            action = dist.sample()
            logp = dist.log_prob(action)
            # logp = F.log_softmax(logit, dim=-1)

            # action = torch.exp(logp).multinomial(num_samples=1).data[0]#logp.max(1)[1].data if args.test else
            state, reward, done, _, _ = env.step(action.numpy()[0])
            state = state.mean(-1)/255
            if config.pomdp: state += np.random.random(state.shape) * 0.3
            state_deque.append(state)
            epr += reward
            reward = np.clip(reward, -1, 1) # reward
            done = done or episode_length >= config.max_frame_episode # don't playing one ep for too long
            
            info['frames'].add_(1) # torch.tensor().add()用完不改变原值，add_()会改变原值
            num_frames = int(info['frames'].item())
            
            if num_frames % config.ckpt_frame == 0: # save every 2M frames
                printlog('\n\t{:.0f}M frames: saved model\n'.format(num_frames/1e6))
                torch.save(shared_model.state_dict(), config.save_dir+'/model.{:.0f}.tar'.format(num_frames/1e6))

            if done: # update shared data
                info['episodes'] += 1
                interp = 1 if info['episodes'][0] == 1 else 1 - config.horizon
                info['run_epr'].mul_(1-interp).add_(interp * epr)
                info['run_loss'].mul_(1-interp).add_(interp * eploss)

            if rank == 0 and time.time() - last_disp_time > 5: # print info ~ every 5 seconds
                elapsed = time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start_time))
                printlog('time {}, episodes {:.0f}, frames {:.6f}M, mean epr {:.2f}, run loss {:.2f}'
                    .format(elapsed, info['episodes'].item(), num_frames/1e6,
                    info['run_epr'].item(), info['run_loss'].item()))
                # print(model.tau_adp_h1, 
                #       model.tau_adp_a, 
                #       model.tau_adp_c, 
                #       model.tau_m_h1, 
                #       model.tau_m_a, 
                #       model.tau_m_c)
        
                last_disp_time = time.time()

            if done:
                episode_length, epr, eploss = 0, 0, 0
                # state = torch.tensor(env.reset()[0])
                
                state_deque = deque(maxlen=config.time_step)
                tmp = env.reset()[0]
                tmp = tmp.mean(-1)/255
                if config.pomdp: tmp += np.random.random(tmp.shape) * 0.3
                for _ in range(config.time_step):
                    state_deque.append(tmp)

            values.append(value) ; logps.append(logp) ; actions.append(action) ; rewards.append(reward)

        next_value = torch.zeros(1,1) if done else model(torch.tensor(state_deque).view(1, config.time_step, 96, 96))[0]
        values.append(next_value.detach())
        loss = cost_func(torch.cat(values), torch.cat(logps), torch.cat(actions), np.asarray(rewards))
        eploss += loss.item()
        shared_optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 40)

        for param, shared_param in zip(model.parameters(), shared_model.parameters()):
            if shared_param.grad is None: shared_param._grad = param.grad # sync gradients with shared model
        shared_optimizer.step()
        
        
    # print(rank,'finish')

if __name__ == "__main__":
    if sys.version_info[0] > 2:
        mp.set_start_method('spawn') # this must not be in global scope
    elif sys.platform == 'linux' or sys.platform == 'linux2':
        raise "Must be using Python 3 with linux!" # or else you get a deadlock in conv2d
    
    if config.render:  config.processes = 1 ; config.test = True # render mode -> test mode w one process
    if config.test:  config.lr = 0 # don't train in render mode
    os.makedirs(config.save_dir) if not os.path.exists(config.save_dir) else None

    torch.manual_seed(config.seed) # 保证所有线程使用同一个mask
    np.random.seed(config.seed)
    mask = (torch.rand(config.hid, config.hid) > config.dropout).int() * (1-torch.eye(config.hid, config.hid)).int()
    masks = [mask]
    for i in range(100):
        if config.dropout_stepping > 0 and (mask==0).sum().item()/config.hid**2 <= config.dropout_stop:
            mask = mask&((torch.rand(config.hid, config.hid) > config.dropout_stepping).int() * (1-torch.eye(config.hid, config.hid)).int())
            masks.append(mask)
        else:
            masks.append(mask)
    shared_model = NNPolicy().share_memory()
    shared_optimizer = SharedAdam(shared_model.parameters(), lr=config.lr)

    info = {k: torch.DoubleTensor([0]).share_memory_() for k in ['run_epr', 'run_loss', 'episodes', 'frames']}
    # info['frames'] += shared_model.try_load(config.save_dir) * 1e6
    if int(info['frames'].item()) == 0: printlog('', end='', mode='w') # clear log file
    
    processes = []
    for rank in range(config.processes):
        p = mp.Process(target=train, args=(shared_model, shared_optimizer, rank, info, masks))
        p.start() ; processes.append(p)
    for p in processes: p.join()
