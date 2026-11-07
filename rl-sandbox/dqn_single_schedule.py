#Library Imports

import pandas as pd
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from gymnasium import spaces
from collections import deque



#Global Constants

TRAINING_ITERATIONS = 1
ACTION_MAP = {0:-1, 1:0, 2:1} 
#This is used to convert indices which need to be used in the replay buffer and tensor qnet into (Buy:-1, Hold:0, Sell:1) to work as a multiplier for the reward function



#Data Import 

dam = pd.read_parquet("..\\data-pull\\combined_data\\dam_data_2023_2026.parquet")
dam["lmp"]


# 1. Custom Gymnasium Environment
class BatteryEnv(gym.Env):
    def __init__(self):
        super().__init__()
        # State space: 25 dimension 
            # First 24 dimensions
                # Continuous vector from -infinity to infinity containing
                # Represents a 24 hour energy price schedule
                # This could be bounded if the training time is too high using a floor/ceiling. 
                # If the price is negative, you'll buy and if the price is really high, you'll sell. We're more interested in the middle range
            # 25th dimension
                # Capacity of the battery 

        low = np.append(np.full(24, -np.inf, dtype=np.float32), 0.0)
        high = np.append(np.full(24, np.inf, dtype=np.float32), 1.0)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        # Action space: -1, 0, 1. Buy: -1, Hold: 0, Sell: 1. Represents the multiplier for the reward.  
            # Mapped by Action Map to get from {0,1,2} to {-1,0,1}
        self.action_space = spaces.Discrete(3)
        self.hours_passed = 0
        self.state = np.append(dam["lmp"][0:24], 0) #Append an empty battery capacity to the 24h schedule to form the state space

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.append(dam["lmp"][0:24],0) #Append an empty battery capacity to the 24h schedule to form the state space
        self.hours_passed = 0
        return self.state, {}

    def get_action_mask(self, state):
        capacity = state[24]
        #print("state: ", state)
        #print("capacity: ", capacity)
        mask = np.array([True, True, True], dtype=bool)
        if capacity == 0:
            mask[2] = False     #Disallow selling if the battery is empty
        elif capacity == 1:
            mask[0] = False     #Disallow buying if the battery is full
        else:
            print("Zack Warning: The capacity was neither 0 nor 1")
        #print(mask)
        return mask


    def step(self, action):
        current_price = self.state[0]
        #print(price_schedule)

        #print(action_schedule)
        actual_action = ACTION_MAP[action]   #Convert from action space {0,1,2} to reward multiplier {-1,0,1}
    
        #Calculate the reward of the decisions we made
        reward = actual_action * current_price
        #print(reward)


        self.hours_passed += 1
        if self.hours_passed + 24 >= len(dam["lmp"]):
            terminated = True   #End the episode when we've reached the end of the data
            truncated = False   #Truncated is only for artifical stoppages, which we will not have
        else:
            terminated = False   #Single step
            truncated = False

        old_capacity = self.state[24]
        new_capacity = old_capacity - actual_action     
            #If the old capacity is 0 (empty), actual action is -1 (buy), new capacity = 0 - (-1) = 1 (full)
            #If old capacity is 1 (full), actual action is 1 (sell), new capacity = 1 - (1) = 0 (empty)
        #print("Old Capacity: ", old_capacity)
        #print("Actual Action: ", actual_action)
        #print("New Capacity: ", new_capacity)

        self.state = np.append(dam["lmp"][self.hours_passed : self.hours_passed+24],new_capacity)   #Move the schedule to the next 24 forecast as well as update the battery capacity
        
        
        return self.state, reward, terminated, truncated, {}


# 2. Q-Network Architecture
class DQN(nn.Module):
    def __init__(self, state_dim=25, action_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim)
        )
        self.state_dim = state_dim
        self.action_dim = action_dim

    def forward(self, x):
        
        out = self.net(x)
        #print("forward 1: ", out)
        #out = out.view(-1, self.state_dim, self.action_dim)
        #print("forward 2: ", out)
        return out


# 3. Environment & Model Initialization
env = BatteryEnv()
q_net = DQN()
target_net = DQN()
target_net.load_state_dict(q_net.state_dict())

optimizer = optim.Adam(q_net.parameters(), lr=1e-3)
memory = deque(maxlen=5000)

gamma = 0.99
epsilon = 1
epsilon_min = 0.01
epsilon_decay = 0.99
batch_size = 32


# 4. Training Loop
for episode in range(TRAINING_ITERATIONS):
    total_reward = 0
    print(f"Episode {episode}/{TRAINING_ITERATIONS}")
    if (episode % 25) == 0:
        pass
        #print(f"Episode {episode}/{TRAINING_ITERATIONS}")
    state, _ = env.reset()
    done = False
    
    while not done:
        state = env.state
        #print(env.hours_passed)
        action_mask = env.get_action_mask(state)
        valid_action_indices = np.where(action_mask)[0]

        # Epsilon-greedy action selection
        #print("Epsilon: ", epsilon)
        if random.random() < epsilon:
            action = np.random.choice(valid_action_indices)
            #print("type 1: ", action)
        else:
            with torch.no_grad():
                state_t = torch.FloatTensor(state)
                #print("state ", state_t)
                #print("qnet ", q_net(state_t))
                #print(q_net(state_t).shape)
                a = state_t
                #print("a: ", a)
                b = q_net(a)
                b[~action_mask] = float('-inf')
                #print("b: ", b)
                c = b.argmax()
                #print("c: ", c)
                d = c.squeeze(0)
                #print("d: ", d)
                e = d.tolist()
                #print("e: ", e)
                action = e
                #print("type 2 ", action)


        next_state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        done = terminated or truncated
        memory.append((state, action, reward, next_state, done))

        # Replay Buffer Sampling & Network Update
        if len(memory) >= batch_size:
            batch = random.sample(memory, batch_size)
            s_arr, a_arr, r_arr, ns_arr, d_arr = zip(*batch)

            states = torch.FloatTensor(np.array(s_arr))
            actions = torch.LongTensor(a_arr).unsqueeze(1)
            rewards = torch.FloatTensor(r_arr)
            next_states = torch.FloatTensor(np.array(ns_arr))
            dones = torch.FloatTensor(d_arr)

            # Compute current Q(s, a)
            z = states
            #print("z: ", z)
            y = q_net(z)
            #print("y: ", y)
            x = y.gather(1,actions)
            w = x.squeeze(1)
            q_values = w
        

            # Target Q-value using Bellman Equation
            with torch.no_grad():
                #print("target_shape: ", target_net(next_states).max(dim=1))
                #print("HELLLOOOOO")
                max_next_q = target_net(next_states).max(dim=1)[0]
                #print("rewards: ",rewards)
                #print("gamma: ", gamma)
                #print("max next q:", max_next_q)
                #print("dones: ", dones)
                targets = rewards + gamma * max_next_q * (1 - dones)

            loss = nn.MSELoss()(q_values, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    if episode % 20 == 0:
        target_net.load_state_dict(q_net.state_dict())
    print("Total Reward: ", total_reward)



