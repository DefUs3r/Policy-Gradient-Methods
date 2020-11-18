'''
Script to implement DDPG taken from https://github.com/facebookresearch/ReAgent
TODO - Test and add to Continuous Action Space (DONE)
'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
import copy

from ActorCriticModels import *
from utilities import *

class DDPG:
    def __init__(self,env,learning_rate = 1e-4,gamma = 0.99):
        self.env = env
        self.state_space = env.observation_space.shape[0]
        self.action_space = env.action_space.shape[0]
        self.action_bound = float(env.action_space.high[0])
        self.device = torch.device("cuda:0")
        self.actor = Actor_DDPG(self.state_space,self.action_space,self.action_bound).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),lr = learning_rate)
        self.critic = Critic_DDPG(self.state_space,self.action_space).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),weight_decay = 1e-2)
        self.gamma = gamma
        self.tau = 0.001
        self.batch_size = 256

    def get_action(self,state):
        state = torch.FloatTensor(state.reshape(1,-1)).to(self.device)
        return self.actor(state).cpu().data.numpy().flatten()

    def update_policy(self,replay_buffer,batch_size):
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
        target_Q = self.critic_target(next_state, self.actor_target(next_state))
        target_Q = reward + (not_done*self.gamma*target_Q).detach()
        current_Q = self.critic(state,action)
        critic_loss = F.mse_loss(current_Q, target_Q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        actor_loss = -self.critic(state,self.actor(state)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        for param, target_param in zip(self.critic.parameters(),self.critic_target.parameters()):
            target_param.data.copy_(self.tau*param.data+(1-self.tau)*target_param.data)

        for param, target_param in zip(self.actor.parameters(),self.actor_target.parameters()):
            target_param.data.copy_(self.tau*param.data+(1-self.tau)*target_param.data)

    def train(self,max_timesteps = 3e4,start_timesteps = 10000):
        replay_buffer = ReplayBuffer(self.state_space,self.action_space)
        state = self.env.reset()
        done = False
        episode_reward = 0
        reward_history = []
        episode_num = 0
        episode_timesteps = 0
        for t in range(1,int(max_timesteps)+1):
            episode_timesteps += 1
            if t < start_timesteps:
                action = self.env.action_space.sample()
            else:
                # self.env.render()
                action = (self.get_action(np.array(state)) + 
                          np.random.normal(0,self.action_bound*0.1,size = self.action_space)).clip(-self.action_bound,self.action_bound)

            next_state, reward, done, _ = self.env.step(action)
            replay_buffer.add(state,action,next_state,reward,float(done))
            state = next_state
            episode_reward += reward
            if t >= start_timesteps:
                self.update_policy(replay_buffer,self.batch_size)

            if done:
                print("\rEpisode: {}, Episode Reward: {}, Average Reward: {}".format(episode_num,episode_reward,np.mean(np.array(reward_history))),end = "")
                state = self.env.reset()
                done = False
                reward_history.append(episode_reward)
                episode_reward = 0
                episode_num += 1
                episode_timesteps = 0

        return reward_history