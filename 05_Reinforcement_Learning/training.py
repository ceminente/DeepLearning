import dill
import numpy as np
import agent
import environment
import matplotlib.pyplot as plt
from tqdm import tqdm
from itertools import product
import pandas as pd

def train_agent(learner, env, alpha, epsilon, episodes, episode_lenght, initial="random", verbose=False):
# perform the training
    rewards_log = []
    for index in range(0, episodes):
        
        
        # start from a random state
        if initial == "random":
            initial = [np.random.randint(0, x), np.random.randint(0, y)]
        # initialize environment
        ##checking if initial and goal state are the same
        bool_init = (initial == env.goal or initial in env.walls)
        while (bool_init):
            if initial == "random":
                initial = [np.random.randint(0, x), np.random.randint(0, y)]
            else: 
                print("Initial state cannot correspond to either goal or walls")
                break
        if bool_init:
            break
        state = initial
        
        #initializes starting point keeping boundaries and goal fixed
        env.initialize_start(initial)
        reward = 0
        # run episode
        for step in range(0, episode_length):
            # find state index
            state_index = state[0] * y + state[1]
            # choose an action
            action = learner.select_action(state_index, epsilon[index]) #epsilon decade tra gli episodi
            # the agent moves in the environment
            result = env.move(action)
            # Q-learning update
            next_index = result[0][0] * y + result[0][1]
            learner.update(state_index, action, result[1], next_index, alpha[index], epsilon[index])
            # update state and reward
            reward += result[1]
            state = result[0]
        reward = reward / episode_length
        rewards_log.append(reward)
        if verbose:
            print('Episode ', index + 1, ': the agent has obtained an average reward of ', reward, ' starting from position ', initial) 
    if verbose:
        fig, ax = plt.subplots(1,1, figsize=(8,7))
        ax.scatter(episodes, rewards_log)
        ax.set_xlabel("Episode", fontsize=15)
        ax.set_ylabel("Reward per step", fontsize=15)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.tick_params(axis='both', which='minor', labelsize=10)
        fig.tight_layout()
        plt.show()
    return rewards_log, learner

def test_agent(learner, env, episode_lenght, initial, plot=True):
    if initial == "random":
        initial = [np.random.randint(0, x), np.random.randint(0, y)]
    # initialize environment
    state = initial
    #initializes starting point keeping boundaries and goal fixed
    env.initialize_start(initial)
    reward = 0
    states_log = [state]
    
    for step in range(0, episode_length):
        # find state index
        state_index = state[0] * y + state[1]
        # choose an action
        action = learner.select_action(state_index, epsilon="greedy") #epsilon è più greedy possibile
        # the agent moves in the environment
        result = env.move(action)
        state = result[0]
        states_log.append(state)
             
    return states_log

def plot_moves(learner, env, states_log, title, moves=True, color_map=True):

    dim = learner.qtable.shape[0]
    
    plt.figure(figsize=(6,6))
    #plot maze
    plt.imshow(env.maze.transpose(), cmap="Greys", origin="lower")
    #plot heatmap of expected reward
    if color_map:
        exp_reward = np.amax(learner.qtable, axis=1)
        plt.imshow(exp_reward.reshape(10,-1).transpose(), origin='lower')
    #plot goal and start
    plt.scatter(env.goal[0], env.goal[1], s=200, c='magenta', marker='o', alpha=0.5) #goal
    plt.scatter(env.start[0], env.start[1], s=200, c='magenta', marker="*", alpha=0.5) #start
    #plot path
    for step in states_log:
        plt.scatter(step[0], step[1], s=100, c= 'magenta', marker=".")

    #plotting the best move for each position
    if moves: 
        exp_reward = np.amax(learner.qtable, axis=1)
        exp_move = np.argwhere(np.vstack((exp_reward,exp_reward,exp_reward,exp_reward,exp_reward)).transpose() == learner.qtable)
        exp_move = np.split(exp_move[:,1], np.unique(exp_move[:, 0], return_index=True)[1][1:])
        for i in range(10):
            for j in range(10):
                if [i,j] in env.walls:
                    pass
                else:
                    plt.text(i, j, exp_move[i*10 + j], ha="left", va="top", color="red")
                
    # Major ticks
    plt.xticks(np.arange(0, 10, 1))
    plt.yticks(np.arange(0, 10, 1))

    # Labels for major ticks
    ax = plt.gca()
    ax.set_xticklabels(np.arange(0, 10, 1))
    ax.set_yticklabels(np.arange(0, 10, 1))

    # Minor ticks
    ax.set_xticks(np.arange(-.5, 10, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 10, 1), minor=True)

    # Gridlines based on minor ticks
    plt.grid(which='minor', color='black', linestyle='-', linewidth=2)
    plt.tight_layout()
    plt.savefig(title)
    plt.show()
    
def epsilon(choice, episodes):
    ei = 0.8
    ef = 0.001
    l = 3
    if choice == "lin":
        return np.linspace(ei, ef, episodes)
    elif "exp":
        return ef + (ei-ef)*np.exp(-l*np.arange(episodes)/episodes)
    





if __name__ == "__main__":

    # FIXED PARAMETERS
    episodes = 2000         # number of training episodes
    episode_length = 50     # maximum episode length
    x = 10                  # horizontal size of the box
    y = 10                  # vertical size of the box
    state = [9,1]           # starting point
    goal = [0,5]            # objective point
    discount = 0.9         # exponential discount factor
    alpha = np.ones(episodes) * 0.25
    eps = epsilon("exp", episodes)
    softmax=True
    sarsa=False

    print("\n")
    print("Training the agent with the following parameters:")
    print("episodes: ", episodes)
    print("episode_length: ", episode_length)
    print("discount: ", discount)
    print("alpha: ", 0.25)
    print("epsilon decay: exp")
    print("Softmax; ", softmax)
    print("SARSA: ", sarsa)
    print("\n")

    #QLEARNING
    walls=[[0,0],[1,1],[2,2,],[4,4],[5,5],[7,7],[8,8],[9,9]]
    swamps = [[9,0],[8,1],[7,2],[5,4],[4,5],[6,3],[2,7],[1,8],[0,9]]
    env = environment.Environment(x, y, state, goal, walls=walls, swamps=swamps)
    learner = agent.Agent(states=(x * y), actions=5, discount=discount, max_reward=1, softmax=softmax, sarsa=sarsa)
    training=True
    if training:
        rewards_log, learner = train_agent(learner, env, alpha, eps, episodes, episode_length, initial=state)
    if training:
        path = test_agent(learner, env, episode_length, initial=state)
        plot_moves(learner, env, path, "path.png", False, False)     