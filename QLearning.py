import gym
import random
import numpy as np
from collections import deque

from collections import defaultdict


EPISODES =  20000
LEARNING_RATE = .1
DISCOUNT_FACTOR = .99
EPSILON = 1
EPSILON_DECAY = .999


def default_Q_value():
    return 0

if __name__ == "__main__":

    random.seed(1)
    np.random.seed(1)
    env = gym.envs.make("FrozenLake-v1")
    env.seed(1)
    env.action_space.np_random.seed(1)


    Q_table = defaultdict(default_Q_value)
    episode_reward_record = deque(maxlen=100)

    for i in range(EPISODES):
        episode_reward = 0
        done = False
        obs = env.reset()

        #biggest value in the row corresponding to a state is the value of that state
        while (not done):
            if random.uniform(0,1) < EPSILON: # take a random action occasionally that isn't necessarily optimal, this will occur more as time goes on due to epsilon decay
                action = env.action_space.sample()
            else:
                prediction = np.array([Q_table[(obs,i)] for i in range(env.action_space.n)]) 
                action =  np.argmax(prediction)

            obs_nextState,reward,done,info = env.step(action)
            episode_reward += reward # update episode reward

            tPlusOne_Q_table = np.array([Q_table[(obs_nextState,i)] for i in range(env.action_space.n)]) #get Q table for state t+1

            # update Q table
            if not done:
                Q_table[obs, action] = (1 - LEARNING_RATE) * (Q_table[obs, action]) + (LEARNING_RATE * (reward + (DISCOUNT_FACTOR * np.max(tPlusOne_Q_table))))
            else:
                Q_table[obs, action] = ((1 - LEARNING_RATE) * (Q_table[obs, action])) + (LEARNING_RATE * reward)
            obs = obs_nextState
            
        EPSILON = EPSILON * EPSILON_DECAY # epsilon decays over time in order to take more random actions and try to find optimal policy

        # record the reward for this episode
        episode_reward_record.append(episode_reward) 

        
        if i%100 ==0 and i>0:
            print("LAST 100 EPISODE AVERAGE REWARD: " + str(sum(list(episode_reward_record))/100))
            print("EPSILON: " + str(EPSILON) )