import numpy as np
import matplotlib.pyplot as plt


class Environment:

    state = []
    goal = []
    boundary = []
    action_map = {
        0: [0, 0], 
        1: [0, 1], 
        2: [0, -1],
        3: [1, 0], 
        4: [-1, 0],
    }
    
    def __init__(self, x, y, initial, goal, walls=[], swamps=[], swamp_reward=-0.5):
        self.boundary = np.asarray([x, y])
        self.start = np.asarray(initial)
        self.state = self.start
        self.goal = goal
        self.walls = walls
        self.swamps = swamps
        self.swamp_reward=swamp_reward
        self.maze = np.zeros((self.boundary))
        for s in self.swamps:
            self.maze[s[0]][s[1]] = 0.5
        for w in self.walls:
            self.maze[w[0]][w[1]] = 1

    # the agent makes an action (0 is stay, 1 is up, 2 is down, 3 is right, 4 is left)
    def move(self, action):
        reward = 0
        movement = self.action_map[action]
        if (action == 0 and (self.state == self.goal).all()):
            reward = 1
        next_state = self.state + np.asarray(movement)
        if(self.check_boundaries(next_state)):
            reward = -1
        elif(list(next_state) in self.walls):
            reward = -1
        else:
            self.state = next_state
        if(list(next_state) in self.swamps):
            reward = self.swamp_reward
        return [self.state, reward]

    # map action index to movement
    def check_boundaries(self, state):
        out = len([num for num in state if num < 0]) #conta quanti elementi di state sono fuori dal margine inferiore
        out += len([num for num in (self.boundary - np.asarray(state)) if num <= 0]) #conta quanti elementi di state sono più grandi di 
                                                                                    #boundary (i.e. fuori)
        return out > 0

    def initialize_start(self, start):
        self.start = np.asarray(start)
        self.state = self.start


    def plot_maze(self):
        fig, ax = plt.subplots(1,1, figsize=(6,6))
        ax.imshow(self.maze.transpose(), cmap="Greys", origin="lower")
        ax.scatter(self.goal[0], self.goal[1], s=200, c='red', marker='o') #goal
        ax.scatter(self.start[0], self.start[1], s=200, c='red', marker="*") #start
        # Major ticks
        ax.set_xticks(np.arange(0, 10, 1))
        ax.set_yticks(np.arange(0, 10, 1))

        # Labels for major ticks
        ax.set_xticklabels(np.arange(0, 10, 1))
        ax.set_yticklabels(np.arange(0, 10, 1))

        # Minor ticks
        ax.set_xticks(np.arange(-.5, 10, 1), minor=True)
        ax.set_yticks(np.arange(-.5, 10, 1), minor=True)

        # Gridlines based on minor ticks
        ax.grid(which='minor', color='black', linestyle='-', linewidth=2)

        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.tick_params(axis='both', which='minor', labelsize=10)

        fig.tight_layout()
        plt.show()