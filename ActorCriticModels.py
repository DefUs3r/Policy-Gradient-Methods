'''
Script to implement Actor Critic Agents and Models
TODO - Test (Done)
TODO - Test and add to Continuous Action Space (Done)
'''

import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.autograd
from torch.autograd import Variable
from utilities import *

#############--------------Vanilla Actor Critic Agent-----------------------#############

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

class ActorCriticCp(nn.Module):
    def __init__(self,state_space,action_space,hidden_size = 128):
        super(ActorCriticCp,self).__init__()
        self.linear1 = nn.Linear(state_space,hidden_size)
        self.actor_linear = nn.Linear(hidden_size,action_space)
        self.critic_linear = nn.Linear(hidden_size,1)
        self.saved_actions = []
        self.rewards = []

    def forward(self,state):

        state = F.relu(self.linear1(state))
        y = F.softmax(self.actor_linear(state),dim = -1)
        x = self.critic_linear(state)

        return x,y
#############-----------------DDPG Actor Critic Agent------------------------#############

class Actor_DDPG(nn.Module):
	def __init__(self,state_space,action_space,action_bound):
		super(Actor_DDPG,self).__init__()

		self.linear1 = nn.Linear(state_space,400)
		self.linear2 = nn.Linear(400,300)
		self.linear3 = nn.Linear(300,action_space)

		self.action_bound = action_bound

	def forward(self,state):
		x = F.relu(self.linear1(state))
		x = F.relu(self.linear2(x))

		return self.action_bound * torch.tanh(self.linear3(x))

class Critic_DDPG(nn.Module):
	def __init__(self,state_space,action_space):
		super(Critic_DDPG,self).__init__()

		self.linear1 = nn.Linear(state_space,400)
		self.linear2 = nn.Linear(400 + action_space,300)
		self.linear3 = nn.Linear(300,1)

	def forward(self,state,action):


		q = F.relu(self.linear1(state))
		q = F.relu(self.linear2(torch.cat([q,action], 1)))

		return self.linear3(q)