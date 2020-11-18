'''
Script containing utility functions
'''
import numpy as np
import gym
from collections import deque
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

#############------------------------Memory Buffer------------------------#############
class ReplayBuffer:
	def __init__(self,state_space,action_space,max_size = int(1e6)):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.state = np.zeros((max_size,state_space))
		self.action = np.zeros((max_size,action_space))
		self.next_state = np.zeros((max_size,state_space))

		self.reward = np.zeros((max_size,1))
		self.not_done = np.zeros((max_size,1))

		self.device = torch.device("cuda:0")

	def add(self,state,action,next_state,reward,done):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward

		self.not_done[self.ptr] = 1. - done


		self.ptr = (self.ptr+1)%self.max_size
		self.size = min(self.size+1,self.max_size)

	def sample(self,batch_size):
		ind = np.random.randint(0,self.size,size = batch_size)

		return (
			torch.FloatTensor(self.state[ind]).to(self.device),
			torch.FloatTensor(self.action[ind]).to(self.device),
			torch.FloatTensor(self.next_state[ind]).to(self.device),
			torch.FloatTensor(self.reward[ind]).to(self.device),
			torch.FloatTensor(self.not_done[ind]).to(self.device),
			)

#############------------------------Tester Functions------------------------#############    
def test(agent,env,render = False):
    state = env.reset()
    r = 0
    done = False
    while not done:
        action = agent.get_action(state)
        new_state,reward,done,_ = env.step(action)
        if render == True:
            env.render()
        state = new_state
        r += np.float64(reward)
    env.close()
    print("\n"+str(r))    
    
#####------Discrete Testers-------#####    
def test_ac_cartpole(agent,env,render = False):
    state = env.reset()
    r = 0
    done = False

    for i in range(200):
        action = agent.get_action(state)
        new_state,reward,done,_ = env.step(action)
        
        if (render == True):
            env.render()
            
        state = new_state
        r += reward
        if done:
            break
            
    env.close()
    print("\n"+str(r))

def test_cp(agent,env,render = False):
    state = env.reset()
    r = 0
    done = False
    for i in range(200):
        action = agent.get_action(state)
        new_state,reward,done,_ = env.step(action[0])
        if render == True:
            env.render()
        state = new_state
        r += reward
        if done:
            break

    env.close()
    print("\n"+str(r))
    
#############------------------------Plotting------------------------#############

#####-------Average Rewards v/s Number of Episodes--------#####
def plot_mean_confInterv(mean, lb, ub, color_mean=None, color_shading=None):
    # plot the shaded range of the confidence intervals
    plt.fill_between(range(mean.shape[0]),ub, lb,
                     color=color_shading, alpha=.5)
    # plot the mean on top
    plt.plot(mean, color_mean)
    
#####-------Returns--------#####    
def return_stats(returns,window_size = 100):
    averaged_returns = np.zeros(len(returns)-window_size+1)
    max_returns = np.zeros(len(returns)-window_size+1)
    min_returns = np.zeros(len(returns)-window_size+1)
    
    for i in range(len(averaged_returns)):
        averaged_returns[i] = np.mean(returns[i:i+window_size])
        max_returns[i] = np.max(returns[i:i+window_size])
        min_returns[i] = np.min(returns[i:i+window_size])
    
    return (averaged_returns,max_returns,min_returns)