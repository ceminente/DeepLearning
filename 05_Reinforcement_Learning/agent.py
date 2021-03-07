import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt

class Agent:
    states = 1
    actions = 1
    discount = 0.9
    max_reward = 1
    qtable = np.matrix([1])
    softmax = False
    sarsa = False
    
    # initialize
    def __init__(self, states, actions, discount, max_reward, softmax, sarsa):
        self.states = states
        self.actions = actions
        self.discount = discount
        self.max_reward = max_reward
        self.softmax = softmax
        self.sarsa = sarsa
        # initialize Q table
        self.qtable = np.ones([states, actions], dtype = float) * max_reward / (1 - discount)
        
    # update function (Sarsa and Q-learning)
    def update(self, state, action, reward, next_state, alpha, epsilon):
        # find the next action (greedy for Q-learning, using the decision policy for Sarsa)
        next_action = self.select_action(next_state, "greedy")
        if (self.sarsa):
            next_action = self.select_action(next_state, epsilon)
        # calculate long-term reward with bootstrap method
        observed = reward + self.discount * self.qtable[next_state, next_action]
        # bootstrap update
        self.qtable[state, action] = self.qtable[state, action] * (1 - alpha) + observed * alpha
        
    # action policy: implements epsilon greedy and softmax
    def select_action(self, state, epsilon):
        qval = self.qtable[state]
        prob = []
        if epsilon == "greedy":
            maxes = np.argwhere(qval==np.amax(qval)).flatten()
            action = np.random.choice(maxes)
        else:
            if (self.softmax):
                # use Softmax distribution
                prob = sp.softmax(qval / epsilon)
            else:
                maxes = np.argwhere(qval==np.amax(qval)).flatten() #just in case there is more than one (otherwise 
                                                                   #argmax just outputs the first encountered index)
                if len(maxes) == self.actions:
                    prob = np.ones(self.actions) / (self.actions)
                else:
                    # assign equal value to all actions
                    prob = np.ones(self.actions) * epsilon / (self.actions - len(maxes))
                    # the best action is taken with probability 1 - epsilon
                    prob[maxes] = (1 - epsilon)/len(maxes)
            action = np.random.choice(range(0, self.actions), p = prob)
        return action

    #computes the maximum values for each row in the q-table (i.e. the most likely states and plots them)
    def get_expected_reward(self, plot):
        exp_reward = np.amax(self.qtable, axis=1)
        dim = self.qtable.shape[0]
        if plot:
            plt.imshow(exp_reward.reshape(int(np.sqrt(dim)),-1))
            plt.show()
        return exp_reward