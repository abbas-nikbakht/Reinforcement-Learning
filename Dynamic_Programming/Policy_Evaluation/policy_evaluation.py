######################  Description #######################

#        Grilled Word 3x4 Dynamic Programing example

############################################################

import gymnasium as gym
import random as random
import numpy as np
# Environment creation.
class GridWorld3x4():
    
    def __init__(self):
        ##
        self.env = gym.make(
            "FrozenLake-v1",
            desc=["SFFG",
                  "FFFH",
                  "FFFF"],
            is_slippery=True,
        )
        
        
        # state 0 
        self.env.unwrapped.P[0][0] = [
            (0.8, 0, -0.04, False), # (prob, next_state, reward, terminated)
            (0.1, 0, -0.04, False),
            (0.1, 4, -0.04, False)
        ]
        
        self.env.unwrapped.P[0][1] = [
            (0.8, 4, -0.04, False), # (prob, next_state, reward, terminated)
            (0.1, 0, -0.04, False),
            (0.1, 1, -0.04, False)
        ]
        
        self.env.unwrapped.P[0][2] = [
            (0.8, 1, -0.04, False), # (prob, next_state, reward, terminated)
            (0.1, 0, -0.04, False),
            (0.1, 4, -0.04, False)
        ]
        
        self.env.unwrapped.P[0][3] = [
            (0.8, 0, -0.04, False), # (prob, next_state, reward, terminated)
            (0.1, 0, -0.04, False),
            (0.1, 1, -0.04, False)
        ]
        
        # state 1 
        self.env.unwrapped.P[1][0] = [
            (0.8, 0, -0.04, False), # (prob, next_state, reward, terminated)
            (0.1, 1, -0.04, False),
            (0.1, 1, -0.04, False)
        ]
        self.env.unwrapped.P[1][1] = [
            (0.8, 1, -0.04, False), # (prob, next_state, reward, terminated)
            (0.1, 0, -0.04, False),
            (0.1, 2, -0.04, False)
        ]
        self.env.unwrapped.P[1][2] = [
            (0.8, 2, -0.04, False), # (prob, next_state, reward, terminated)
            (0.1, 1, -0.04, False),
            (0.1, 1, -0.04, False)
        ]
        self.env.unwrapped.P[1][3] = [
            (0.8, 1, -0.04, False), # (prob, next_state, reward, terminated)
            (0.1, 0, -0.04, False),
            (0.1, 2, -0.04, False)
        ]
        
        # state 2 
        self.env.unwrapped.P[2][0] = [
            (0.8, 1, -0.04, False), # (prob, next_state, reward, terminated)
            (0.1, 2, -0.04, False),
            (0.1, 6, -0.04, False)
        ]
        self.env.unwrapped.P[2][1] = [
            (0.8, 6, -0.04, False), # (prob, next_state, reward, terminated)
            (0.1, 1, -0.04, False),
            (0.1, 3, 1-0.04, True)
        ]
        self.env.unwrapped.P[2][2] = [
            (0.8, 3, 1-0.04, True), # (prob, next_state, reward, terminated)
            (0.1, 2, -0.04, False),
            (0.1, 6, -0.04, False)
        ]
        self.env.unwrapped.P[2][3] = [
            (0.8, 2, -0.04, False), # (prob, next_state, reward, terminated)
            (0.1, 3, 1-0.04, True),
            (0.1, 1, -0.04, False)
        ]
        
        # state 3 
         # Terminal
         
        # state 4 
        self.env.unwrapped.P[4][0] = [
            (0.8, 4, -0.04, False), # (prob, next_state, reward, terminated)
            (0.1, 0, -0.04, False),
            (0.1, 8, -0.04, False)
        ]
        self.env.unwrapped.P[4][1] = [
            (0.8, 8, -0.04, False), # (prob, next_state, reward, terminated)
            (0.1, 4, -0.04, False),
            (0.1, 4, -0.04, False)
        ]
        self.env.unwrapped.P[4][2] = [
            (0.8, 4, -0.04, False), # (prob, next_state, reward, terminated)
            (0.1, 0, -0.04, False),
            (0.1, 8, -0.04, False)
        ]
        self.env.unwrapped.P[4][3] = [
            (0.8, 0, -0.04, False), # (prob, next_state, reward, terminated)
            (0.1, 4, -0.04, False),
            (0.1, 4, -0.04, False)
        ]
        
        # state 5 
            # Wall
            
        # state 6 
        self.env.unwrapped.P[6][0] = [
            (0.8, 6, -0.04, False), # (prob, next_state, reward, terminated)
            (0.1, 2, -0.04, False),
            (0.1, 10, -0.04, False)
        ]
        self.env.unwrapped.P[6][1] = [
            (0.8, 10, -0.04, False), # (prob, next_state, reward, terminated)
            (0.1, 6, -0.04, False),
            (0.1, 7, -1-0.04, True)
        ]
        self.env.unwrapped.P[6][2] = [
            (0.8, 7, -1-0.04, True), # (prob, next_state, reward, terminated)
            (0.1, 2, -0.04, False),
            (0.1, 10, -0.04, False)
        ]
        self.env.unwrapped.P[6][3] = [
            (0.8, 2, -0.04, False), # (prob, next_state, reward, terminated)
            (0.1, 6, -0.04, False),
            (0.1, 7, -1-0.04, True)
        ]
        
        # state 7 
            # Terminal
            
        # state 8 
        self.env.unwrapped.P[8][0] = [
            (0.8, 8, -0.04, False), # (prob, next_state, reward, terminated)
            (0.1, 4, -0.04, False),
            (0.1, 8, -0.04, False)
        ]
        self.env.unwrapped.P[8][1] = [
            (0.8, 8, -0.04, False), # (prob, next_state, reward, terminated)
            (0.1, 8, -0.04, False),
            (0.1, 9, -0.04, False)
        ]
        self.env.unwrapped.P[8][2] = [
            (0.8, 9, -0.04, False), # (prob, next_state, reward, terminated)
            (0.1, 4, -0.04, False),
            (0.1, 8, -0.04, False)
        ]
        self.env.unwrapped.P[8][3] = [
            (0.8, 4, -0.04, False), # (prob, next_state, reward, terminated)
            (0.1, 9, -0.04, False),
            (0.1, 8, -0.04, False)
        ]
        
        # state 9 
        self.env.unwrapped.P[9][0] = [
            (0.8, 8, -0.04, False), # (prob, next_state, reward, terminated)
            (0.1, 9, -0.04, False),
            (0.1, 9, -0.04, False)
        ]
        self.env.unwrapped.P[9][1] = [
            (0.8, 9, -0.04, False), # (prob, next_state, reward, terminated)
            (0.1, 8, -0.04, False),
            (0.1, 10, -0.04, False)
        ]
        self.env.unwrapped.P[9][2] = [
            (0.8, 10, -0.04, False), # (prob, next_state, reward, terminated)
            (0.1, 9, -0.04, False),
            (0.1, 9, -0.04, False)
        ]
        self.env.unwrapped.P[9][3] = [
            (0.8, 9, -0.04, False), # (prob, next_state, reward, terminated)
            (0.1, 8, -0.04, False),
            (0.1, 10, -0.04, False)
        ]
        
        # state 10 
        self.env.unwrapped.P[10][0] = [
            (0.8, 9, -0.04, False), # (prob, next_state, reward, terminated)
            (0.1, 6, -0.04, False),
            (0.1, 10, -0.04, False)
        ]
        self.env.unwrapped.P[10][1] = [
            (0.8, 10, -0.04, False), # (prob, next_state, reward, terminated)
            (0.1, 9, -0.04, False),
            (0.1, 11, -0.04, False)
        ]
        self.env.unwrapped.P[10][2] = [
            (0.8, 11, -0.04, False), # (prob, next_state, reward, terminated)
            (0.1, 6, -0.04, False),
            (0.1, 10, -0.04, False)
        ]
        self.env.unwrapped.P[10][3] = [
            (0.8, 6, -0.04, False), # (prob, next_state, reward, terminated)
            (0.1, 11, -0.04, False),
            (0.1, 9, -0.04, False)
        ]
        
        # state 11 
        self.env.unwrapped.P[11][0] = [
            (0.8, 10, -0.04, False), # (prob, next_state, reward, terminated)
            (0.1, 7, -1-0.04, True),
            (0.1, 11, -0.04, False)
        ]
        self.env.unwrapped.P[11][1] = [
            (0.8, 11, -0.04, False), # (prob, next_state, reward, terminated)
            (0.1, 10, -0.04, False),
            (0.1, 11, -0.04, False)
        ]
        self.env.unwrapped.P[11][2] = [
            (0.8, 11, -0.04, False), # (prob, next_state, reward, terminated)
            (0.1, 7, -1-0.04, True),
            (0.1, 11, -0.04, False)
        ]
        self.env.unwrapped.P[11][3] = [
            (0.8, 7, -1-0.04, True), # (prob, next_state, reward, terminated)
            (0.1, 10, -0.04, False),
            (0.1, 11, -0.04, False)
        ]
        
        
        
    def step(self, state_initial, action): 
        ### state_inital
        self.env.reset()
        
        base = self.env.unwrapped
        base.s = state_initial 
        
        state, reward, terminated, truncated, info = self.env.step(action)

        return state, reward, terminated, truncated, info

    def reset(self,seed):
        self.env.reset(seed=seed)
        


# policy(state,action)
policy_pi= {
    (0,0): 0, (0,1): 0, (0,2): 1,(0,3): 0,
    (1,0): 0, (1,1): 0, (1,2): 1,(1,3): 0,
    (2,0): 0, (2,1): 0, (2,2): 1,(2,3): 0,
    (4,0): 0, (4,1): 0, (4,2): 0,(4,3): 1,
    (6,0): 0, (6,1): 0, (6,2): 1,(6,3): 0,
    (8,0): 0, (8,1): 0, (8,2): 0,(8,3): 1,
    (9,0): 0, (9,1): 0, (9,2): 1,(9,3): 0,
    (10,0): 0,(10,1): 0,(10,2): 0,(10,3): 1,
    (11,0): 0,(11,1): 0,(11,2): 0,(11,3): 1,
}

v = {
    (0): 0,
    (1): 0,
    (2): 0,
    (3): 0,
    (4): 0,
    (5): 0,
    (6): 0,
    (7): 0,
    (8): 0,
    (9): 0,
    (10): 0,
    (11): 0, 
} 

########################## Parameters #####################
# enviroment
EnviGridWorld3x4 = GridWorld3x4()

P=EnviGridWorld3x4.env.unwrapped.P    

# episode_all= 5000


### seed = controls randomness
# To make it completely random, disable these two lines below.

# EnviGridWorld3x4.reset(seed=3)
# random.seed(3)

##########################################################

for _ in range(10000):      # k: for Jacobi method 
    for s in [0,1,2,4,6,8,9,10,11]:
        
        v[(s)]=0
        
        for a in [0,1,2,3]:
            
            for prob, s_next, reward, done in P[s][a]:
    
                # print(prob, s_next, reward, done)
        
                v[(s)]= v[(s)]+policy_pi[(s,a)]*prob*(reward+v[(s_next)])
        
        
        
        
    
    
    
    