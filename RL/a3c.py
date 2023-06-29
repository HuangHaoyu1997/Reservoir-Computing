# Baby Advantage Actor-Critic | Sam Greydanus | October 2017 | MIT License

from __future__ import print_function
import torch, os, gym, time, glob, argparse, sys
import numpy as np
from scipy.signal import lfilter
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
os.environ['OMP_NUM_THREADS'] = '1'

class config:
    env = 'LunarLander-v2'
    processes = 4
    render = False
    test = False
    update_freq = 20
    lr = 1e-4
    seed = 1
    gamma = 0.99
    tau = 1.0
    max_frame = 1e7
    max_frame_episode = 1e4
    ckpt_frame = 2e6
    horizon = 0.99 # moving avg
    
    
    save_dir = './log/{}-'.format(env.lower()) # keep the directory structure simple
    datetime = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time()))[5:]
    input = 8
    hid = 64
    output = gym.make(env).action_space.n

discount = lambda x, gamma: lfilter([1],[1,-gamma],x[::-1])[::-1] # discounted rewards one liner

def printlog(s, end='\n', mode='a'):
    print(s, end=end) ; f=open(config.save_dir+config.datetime+'.txt',mode) ; f.write(s+'\n') ; f.close()

class NNPolicy(nn.Module): # an actor-critic neural network
    def __init__(self):
        super(NNPolicy, self).__init__()
        self.fc1 = nn.Linear(config.input, config.hid)
        self.critic_linear, self.actor_linear = nn.Linear(config.hid, 1), nn.Linear(config.hid, config.output)

    def forward(self, inputs):
        x = F.elu(self.fc1(inputs))
        return self.critic_linear(x), self.actor_linear(x)
    
    def try_load(self, save_dir):
        paths = glob.glob(save_dir + '*.tar') ; step = 0
        if len(paths) > 0:
            ckpts = [int(s.split('.')[-2]) for s in paths]
            ix = np.argmax(ckpts) ; step = ckpts[ix]
            self.load_state_dict(torch.load(paths[ix]))
        print("\tno saved models") if step is 0 else print("\tloaded model: {}".format(paths[ix]))
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
    logpys = logps.gather(1, torch.tensor(actions).view(-1,1))
    gen_adv_est = discount(delta_t, config.gamma * config.tau)
    policy_loss = -(logpys.view(-1) * torch.FloatTensor(gen_adv_est.copy())).sum()
    
    # l2 loss over value estimator
    rewards[-1] += config.gamma * np_values[-1]
    discounted_r = discount(np.asarray(rewards), config.gamma)
    discounted_r = torch.tensor(discounted_r.copy(), dtype=torch.float32)
    value_loss = .5 * (discounted_r - values[:-1,0]).pow(2).sum()

    entropy_loss = (-logps * torch.exp(logps)).sum() # entropy definition, for entropy regularization
    return policy_loss + 0.5 * value_loss - 0.01 * entropy_loss

def train(shared_model, shared_optimizer, rank, info):
    print(rank,'begin')
    env = gym.make(config.env) # make a local (unshared) environment
    env.reset(seed=config.seed + rank) ; torch.manual_seed(config.seed + rank) # seed everything
    model = NNPolicy() # a local/unshared model
    state = torch.tensor(env.reset()) # get first state

    start_time = last_disp_time = time.time()
    episode_length, epr, eploss, done  = 0, 0, 0, True # bookkeeping

    while info['frames'][0] <= config.max_frame or config.test: # openai baselines uses 40M frames...we'll use 80M
        model.load_state_dict(shared_model.state_dict()) # sync with main shared model

        values, logps, actions, rewards = [], [], [], [] # save values for computing gradientss

        for step in range(config.update_freq):
            episode_length += 1
            value, logit = model(state)
            logp = F.log_softmax(logit, dim=-1)

            action = torch.exp(logp).multinomial(num_samples=1).data[0]#logp.max(1)[1].data if args.test else
            state, reward, done, _, _ = env.step(action.numpy())
            if config.render: env.render()

            state = torch.tensor(state)
            epr += reward
            reward = np.clip(reward, -1, 1) # reward
            done = done or episode_length >= config.max_frame_episode # don't playing one ep for too long
            
            info['frames'].add_(1) # torch.tensor().add()用完不改变原值，add_()会改变原值
            num_frames = int(info['frames'].item())
            
            if num_frames % config.ckpt_frame == 0: # save every 2M frames
                printlog('\n\t{:.0f}M frames: saved model\n'.format(num_frames/1e6))
                torch.save(shared_model.state_dict(), config.save_dir+'model.{:.0f}.tar'.format(num_frames/1e6))

            if done: # update shared data
                info['episodes'] += 1
                interp = 1 if info['episodes'][0] == 1 else 1 - config.horizon
                info['run_epr'].mul_(1-interp).add_(interp * epr)
                info['run_loss'].mul_(1-interp).add_(interp * eploss)

            if rank == 0 and time.time() - last_disp_time > 60: # print info ~ every minute
                elapsed = time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start_time))
                printlog('time {}, episodes {:.0f}, frames {:.1f}M, mean epr {:.2f}, run loss {:.2f}'
                    .format(elapsed, info['episodes'].item(), num_frames/1e6,
                    info['run_epr'].item(), info['run_loss'].item()))
                last_disp_time = time.time()

            if done: # maybe print info.
                episode_length, epr, eploss = 0, 0, 0
                state = torch.tensor(env.reset()[0])

            values.append(value.unsqueeze(0)) ; logps.append(logp.unsqueeze(0)) ; actions.append(action.unsqueeze(0)) ; rewards.append(reward)

        next_value = torch.zeros(1,1) if done else model(state.unsqueeze(0))[0]
        values.append(next_value.detach())

        loss = cost_func(torch.cat(values), torch.cat(logps), torch.cat(actions), np.asarray(rewards))
        eploss += loss.item()
        shared_optimizer.zero_grad() ; loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 40)

        for param, shared_param in zip(model.parameters(), shared_model.parameters()):
            if shared_param.grad is None: shared_param._grad = param.grad # sync gradients with shared model
        shared_optimizer.step()
    print(rank,'finish')


if __name__ == "__main__":
    if sys.version_info[0] > 2:
        mp.set_start_method('spawn') # this must not be in global scope
    elif sys.platform == 'linux' or sys.platform == 'linux2':
        raise "Must be using Python 3 with linux!" # or else you get a deadlock in conv2d
    
    if config.render:  config.processes = 1 ; config.test = True # render mode -> test mode w one process
    if config.test:  config.lr = 0 # don't train in render mode
    os.makedirs(config.save_dir) if not os.path.exists(config.save_dir) else None # make dir to save models etc.

    torch.manual_seed(config.seed)
    shared_model = NNPolicy().share_memory()
    shared_optimizer = SharedAdam(shared_model.parameters(), lr=config.lr)

    info = {k: torch.DoubleTensor([0]).share_memory_() for k in ['run_epr', 'run_loss', 'episodes', 'frames']}
    info['frames'] += shared_model.try_load(config.save_dir) * 1e6
    if int(info['frames'].item()) == 0: printlog('', end='', mode='w') # clear log file
    
    processes = []
    for rank in range(config.processes):
        p = mp.Process(target=train, args=(shared_model, shared_optimizer, rank, info))
        p.start() ; processes.append(p)
    for p in processes: p.join()
