import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import math
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

from torch.utils.tensorboard import SummaryWriter
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

if __name__ == '__main__':

    class config:
        env_name = 'CartPole-v1'
        input = gym.make(env_name).observation_space.shape[0]
        output = gym.make(env_name).action_space.n
        
        env_name = 'LunarLander-v2'
        input = gym.make(env_name).observation_space.shape[0]
        output = gym.make(env_name).action_space.n
        
        seed = 2
        num_envs = 4
        
        std = 0
        hid = 128
        num_steps = 128
        max_steps = 100000
        batch = 256
        ppo_epochs = 30
        time_step = 16  # Number of steps to unroll
        lr = 5e-4
        entropy_loss = 1e-4
        b_j0 = 0.01  # neural threshold baseline
        R_m = 1 # membrane resistance
        dt = 1

        gamma = 0.5  # gradient scale
        gradient_type = 'MG' # 'G', 'slayer', 'linear' 窗型函数
        scale = 6.        # special for 'MG'
        hight = 0.15      # special for 'MG'
        lens = 0.5  # hyper-parameters of approximate function
        trials = 5
        dropout = 0
        dropout_stepping = 0.01
        dropout_stop = 0.90
        smoothing = 0.1
        device = torch.device('cuda')

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    torch.manual_seed(config.seed)
    torch.backends.cudnn.deterministic = True

    def make_env():
        def _thunk():
            env = gym.make(config.env_name)
            # env.seed(seed)
            env.reset(seed=config.seed)
            return env
        return _thunk

    envs = [make_env() for i in range(config.num_envs)]
    envs = SubprocVecEnv(envs)
    env = gym.make(config.env_name)
    # env.seed(seed)
    env.reset(seed=config.seed)

    def gaussian(x, mu=0., sigma=.5):
        return torch.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / torch.sqrt(2 * torch.tensor(torch.pi)) / sigma

    class ActFun_adp(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input):  # input = membrane potential- threshold
            ctx.save_for_backward(input)
            return input.gt(0).float()  # is firing ???

        @staticmethod
        def backward(ctx, grad_output):  # approximate the gradients
            input, = ctx.saved_tensors
            grad_input = grad_output.clone()
            if config.gradient_type == 'G':
                temp = torch.exp(-(input**2)/(2*config.lens**2))/torch.sqrt(2*torch.tensor(torch.pi))/config.lens
            elif config.gradient_type == 'MG':
                temp = gaussian(input, mu=0., sigma=config.lens) * (1. + config.hight) \
                    - gaussian(input, mu=config.lens, sigma=config.scale * config.lens) * config.hight \
                    - gaussian(input, mu=-config.lens, sigma=config.scale * config.lens) * config.hight
            elif config.gradient_type =='linear':
                temp = F.relu(1-input.abs())
            elif config.gradient_type == 'slayer':
                temp = torch.exp(-5*input.abs())
            return grad_input * temp.float() * config.gamma

    act_fun_adp = ActFun_adp.apply

    class SNN(nn.Module):
        def __init__(self):
            super(SNN, self).__init__()
            input = config.input
            hid = config.hid
            out = config.output
            self.inpt_hid1 = nn.Linear(input, hid)
            self.hid1_critic = nn.Linear(hid, 1)
            self.hid1_actor = nn.Linear(hid, out)
            
            self.log_std = nn.Parameter(torch.ones(1, config.output) * config.std)
            
            nn.init.xavier_uniform_(self.inpt_hid1.weight) # 保持输入输出的方差一致，避免所有输出值都趋向于0。通用方法，适用于任何激活函数
            nn.init.xavier_uniform_(self.hid1_critic.weight)
            nn.init.xavier_uniform_(self.hid1_actor.weight)

            nn.init.constant_(self.inpt_hid1.bias, 0)
            nn.init.constant_(self.hid1_critic.bias, 0)
            nn.init.constant_(self.hid1_actor.bias, 0)
            
            self.tau_adp_h1 = nn.Parameter(torch.Tensor(hid))
            self.tau_adp_a = nn.Parameter(torch.Tensor(out))
            self.tau_adp_c = nn.Parameter(torch.Tensor(1))
            
            self.tau_m_h1 = nn.Parameter(torch.Tensor(hid))
            self.tau_m_a = nn.Parameter(torch.Tensor(out))
            self.tau_m_c = nn.Parameter(torch.Tensor(1))

            nn.init.normal_(self.tau_adp_h1, 80, 10)
            nn.init.normal_(self.tau_adp_a, 80, 10)
            nn.init.normal_(self.tau_adp_c, 80, 10)
            nn.init.normal_(self.tau_m_h1, 10., 5)
            nn.init.normal_(self.tau_m_a, 10., 5)
            nn.init.normal_(self.tau_m_c, 10., 5)

            self.dp = nn.Dropout(0.1)
            self.b_hid1 = 0
        
        def output_Neuron(self, inputs, mem, tau_m, dt=1):
            """The read out neuron is leaky integrator without spike"""
            # alpha = torch.exp(-1. * dt / torch.FloatTensor([30.])).cuda()
            alpha = torch.exp(-1. * dt / tau_m).cuda()
            mem = mem * alpha + (1. - alpha) * config.R_m * inputs
            return mem
        
        def mem_update_adp(self, inputs, mem, spike, tau_adp, b, tau_m, dt=1):
            alpha = torch.exp(-1. * dt / tau_m).cuda()
            ro = torch.exp(-1. * dt / tau_adp).cuda()
            b = ro * b + (1 - ro) * spike
            B = config.b_j0 + 1.8 * b
            mem = mem * alpha + (1 - alpha) * config.R_m * inputs - B * spike * dt
            spike = act_fun_adp(mem - B)
            return mem, spike, B, b
        
        def forward(self, input):
            batch, _ = input.shape
            self.b_hid1 = config.b_j0
            hid1_mem = torch.rand(batch, config.hid).to(config.device)
            hid1_spk = torch.zeros(batch, config.hid).to(config.device)
            
            a_out_mem = torch.rand(batch, config.output).to(config.device)
            c_out_mem = torch.rand(batch, 1).to(config.device)
            value = torch.zeros(batch, 1).to(config.device)
            action = torch.zeros(batch, config.output).to(config.device)

            # x.shape = [4,4] inference or [batch, 4] training
            # 持续同一个强度输入
            for _ in range(config.time_step):
                ########## Layer 1 ##########
                inpt_hid1 = self.inpt_hid1(input.float())
                hid1_mem, hid1_spk, theta_h1, self.b_hid1 = self.mem_update_adp(inpt_hid1, hid1_mem, hid1_spk,
                                                                                self.tau_adp_h1, self.b_hid1, self.tau_m_h1)
                # hid1_spk = self.dp(hid1_spk)
                ########## Layer out ##########
                inpt_actor = self.hid1_actor(hid1_spk)
                inpt_critic = self.hid1_critic(hid1_spk)
                c_out_mem = self.output_Neuron(inpt_critic, c_out_mem, self.tau_m_c)
                a_out_mem = self.output_Neuron(inpt_actor, a_out_mem, self.tau_m_a)
                
                value += F.softmax(c_out_mem, dim=1)
                action += F.softmax(a_out_mem, dim=1)
            std   = self.log_std.exp().expand_as(action)
            dist  = Normal(action, std)
            return dist, value
        

    def test_env(model, env, vis=False):
        state, info = env.reset()
        if vis: env.render()
        done = False
        total_reward = 0
        while not done:
            state = torch.FloatTensor(state).unsqueeze(0).to(config.device)
            dist, _ = model(state)
            next_state, reward, done, _, _ = env.step(torch.max(dist.sample(), 1)[1].cpu().numpy()[0])
            state = next_state
            if vis: env.render()
            total_reward += reward
        return total_reward

    # GAE
    def compute_gae(next_value, rewards, masks, values, gamma=0.99, tau=0.95):
        values = values + [next_value]
        gae = 0
        returns = []
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
            gae = delta + gamma * tau * masks[step] * gae
            returns.insert(0, gae + values[step])
        return returns


    # Proximal Policy Optimization Algorithm
    # Arxiv: "https://arxiv.org/abs/1707.06347"
    def ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantage):
        batch_size = states.size(0)
        ids = np.random.permutation(batch_size)
        ids = np.split(ids[:batch_size // mini_batch_size * mini_batch_size], batch_size // mini_batch_size)
        for i in range(len(ids)):
            yield states[ids[i], :], actions[ids[i], :], log_probs[ids[i], :], returns[ids[i], :], advantage[ids[i], :]

    def ppo_update(model, optimizer, ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantages, clip_param=0.2):
        for _ in range(ppo_epochs):
            for state, action, old_log_probs, return_, advantage in ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantages):
                dist, value = model(state)
                entropy = dist.entropy().mean()
                new_log_probs = dist.log_prob(action)

                ratio = (new_log_probs - old_log_probs).exp()
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage

                actor_loss  = - torch.min(surr1, surr2).mean()
                critic_loss = (return_ - value).pow(2).mean()

                loss = 0.5 * critic_loss + actor_loss - config.entropy_loss * entropy

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    datetime = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time()))[5:]
    model = SNN().to(config.device)
    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    step_idx  = 0
    state = envs.reset()
    writer = SummaryWriter(log_dir='./log')
    while step_idx < config.max_steps:

        log_probs = []
        values    = []
        states    = []
        actions   = []
        rewards   = []
        masks     = []
        entropy = 0

        for _ in range(config.num_steps):
            state = torch.FloatTensor(state).to(config.device)
            dist, value = model(state)

            action = dist.sample()
            next_state, reward, done, _ = envs.step(torch.max(action, 1)[1].cpu().numpy())

            log_prob = dist.log_prob(action)
            entropy += dist.entropy().mean()

            log_probs.append(log_prob)
            values.append(value)
            rewards.append(torch.FloatTensor(reward).unsqueeze(1).to(config.device))
            masks.append(torch.FloatTensor(1 - done).unsqueeze(1).to(config.device))

            states.append(state)
            actions.append(action)

            state = next_state
            step_idx += 1

            if step_idx % 100 == 0:
                test_reward = test_env(model, env)
                print('Step: %d, Reward: %.2f' % (step_idx, test_reward))
                writer.add_scalar('SNN-PPO-'+ datetime + config.env_name + '/Reward', test_reward, step_idx)

        next_state = torch.FloatTensor(next_state).to(config.device)
        _, next_value = model(next_state)

        returns = compute_gae(next_value, rewards, masks, values)

        returns   = torch.cat(returns).detach()
        log_probs = torch.cat(log_probs).detach()
        values    = torch.cat(values).detach()
        states    = torch.cat(states)
        actions   = torch.cat(actions)
        advantage = returns - values

        ppo_update(model, optimizer, config.ppo_epochs, config.batch, states, actions, log_probs, returns, advantage)

    print('----------------------------')
    print('Complete')

    writer.close()