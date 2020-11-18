'''
Script to implement Agents for CartPole-v0
TODO - Test and add to Discrete Action Space (Done)
'''

import sys
import torch  
import gym
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

from ActorCriticModels import *
from utilities import *

class Vanilla_Policy_Gradient(nn.Module):
    def __init__(self,state_space,action_space,hidden_size = 256):
        super(Vanilla_Policy_Gradient, self).__init__()
        self.linear1 = nn.Linear(state_space,hidden_size)
        self.linear2 = nn.Linear(hidden_size,action_space)

    def forward(self,state):
        x = F.relu(self.linear1(state))
        x = F.softmax(self.linear2(x),dim = 1)

        return x

class ActorCritic(nn.Module):
    def __init__(self,state_space,action_space,hidden_size = 256):
        super(ActorCritic,self).__init__()
        self.critic_linear1 = nn.Linear(state_space,hidden_size)
        self.critic_linear2 = nn.Linear(hidden_size,1)
        self.actor_linear1 = nn.Linear(state_space,hidden_size)
        self.actor_linear2 = nn.Linear(hidden_size,action_space)

    def forward(self,state):
        x = F.relu(self.critic_linear1(state))
        x = self.critic_linear2(x)
        y = F.relu(self.actor_linear1(state))
        y = F.softmax(self.actor_linear2(y),dim = 1)
        return x,y

class CartPole_agent:
    def __init__(self,env,learning_rate = 3e-4,gamma = 0.99):
        self.env = env
        self.action_space = env.action_space.n
        self.state_space = env.observation_space.shape[0]
        self.gamma = gamma
        self.policy = Vanilla_Policy_Gradient(self.state_space,self.action_space)
        self.optimizer = optim.Adam(self.policy.parameters(),lr = learning_rate)
        self.var_reward = []
        self.loss = []

    def get_action(self,state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.policy.forward(Variable(state))
        action = np.random.choice(self.action_space, p = np.squeeze(probs.detach().numpy()))
        return action,probs

    def update_policy(self,rewards,log_probs,baseline):
        discounted_rewards = []
        for t in range(len(rewards)):
            Gt = 0
            pw = 0
            for r in rewards[t:]:
                Gt = Gt + self.gamma**pw * r
                pw = pw + 1
            discounted_rewards.append(Gt)

        discounted_rewards = torch.FloatTensor(discounted_rewards)
        if baseline == True:
            discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9) # normalize discounted rewards
        self.var_reward.append((discounted_rewards.sum()).item())

        policy_gradient = []
        for log_prob, Gt in zip(log_probs, discounted_rewards):
            policy_gradient.append(-log_prob * Gt)

        self.optimizer.zero_grad()
        policy_gradient = torch.stack(policy_gradient).sum()
        self.loss.append(policy_gradient)
        policy_gradient.backward()
        self.optimizer.step()

    def train(self, max_episode=3000,baseline = False):
        reward_history = []
        for episode in range(1,max_episode+1):
            state = self.env.reset()
            log_probs = []
            rewards = []
            episode_reward = 0
            done = False

            while not done:
                action, probs = self.get_action(state)
                log_prob = torch.log(probs.squeeze(0)[action])
                new_state, reward, done, _ = self.env.step(action)

                log_probs.append(log_prob)
                rewards.append(reward)
                episode_reward += reward                
                state = new_state

            self.update_policy(rewards,log_probs,baseline)

            if episode != 1:
                print("\rEpisode: {}, Episode Reward: {}, Average Reward: {}".format(
                    episode,
                    episode_reward,
                    np.mean(np.array(reward_history[episode-100:episode-1]))),end = "")
                sys.stdout.flush()

            reward_history.append(episode_reward)

        return reward_history

class ActorCritic_agent():

    def __init__(self,env,learning_rate = 3e-2,gamma = 0.99):
        self.env = env
        self.action_space = env.action_space.n
        self.state_space = env.observation_space.shape[0]
        self.gamma = gamma
        self.policy = ActorCriticCp(self.state_space,self.action_space)
        self.SavedAction = namedtuple('SavedAction',['log_prob','value'])
        self.optimizer = optim.Adam(self.policy.parameters(),lr = learning_rate)

    def get_action(self,state):
        state = torch.from_numpy(state).float()
        value , probs = self.policy(state)
        action = np.random.choice(self.action_space, p = np.squeeze(probs.detach().numpy()))
        self.policy.saved_actions.append(self.SavedAction(torch.log(probs.squeeze(0)[action]),value))
        return action

    def update_policy(self):
        R = 0
        saved_actions = self.policy.saved_actions
        policy_losses = []
        value_losses = []
        returns = []
        for r in self.policy.rewards[::-1]:
            R = r+self.gamma*R
            returns.insert(0,R)

        returns = torch.tensor(returns)
        returns = (returns - returns.mean())/(returns.std()+1e-9)
        for (log_prob,value), R in zip(saved_actions,returns):
            advantage = R - value.item()
            policy_losses.append(-log_prob*advantage)
            value_losses.append(F.smooth_l1_loss(value,torch.tensor([R])))

        self.optimizer.zero_grad()
        loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
        loss.backward()
        self.optimizer.step()
        del self.policy.rewards[:]
        del self.policy.saved_actions[:]

    def train(self,max_episode = 3000):
        reward_history = []
        for episode in range(1,max_episode+1):
            state = self.env.reset()
            episode_reward = 0
            done = False
            while not done:
                action = self.get_action(state)
                state,reward,done,_ = self.env.step(action)
                self.policy.rewards.append(reward)
                episode_reward += reward
            self.update_policy()

            if episode != 1:
                print("\rEpisode: {}, Episode Reward: {}, Average Reward: {}".format(
                    episode,
                    episode_reward,
                    np.mean(np.array(reward_history[episode-100:episode-1]))),end = "")
                sys.stdout.flush()

            reward_history.append(episode_reward)

        return reward_history
    
# env = gym.make('CartPole-v0')
# agent = ActorCritic_agent(env)
# window_size = 100
# reward_history=agent.train(2000)

# avg,max_returns,min_returns = return_stats(reward_history,window_size)
# plot_mean_confInterv(avg,min_returns,max_returns,'r','r')
# plt.show()