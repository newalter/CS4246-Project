import random

class QAgent():
    actionSpace = None
    totalMoves = 0
    previousAction = None
    previousObservation = None
    previousReward = None
    alpha = 0.5
    gamma = 0.9
    Q = {} # table for Q-function
    frequency = {}
    def __init__(self, actionSpace):
        self.actionSpace = actionSpace

    def move(self, observation, reward, done):
        if done:
            self.Q[(observation[0],observation[1])] = reward
        s_a = (self.previousObservation[0], self.previousObservation[1], self.previousAction)
        self.frequency[s_a] = self.frequency.get(s_a, 0) + 1
        this_q = self.Q.get(s_a, 0)
        self.Q[s_a] = this_q + self.alpha*(self.previousReward + self.gamma * self.max_lookup(observation) - this_q)
        self.previousObservation = observation
        self.previousReward = reward
        action = self.get_action(observation)
        self.previousAction = action
        return action

    def new_episode(self, observation):
        action = self.get_action(observation)
        self.previousAction = action
        self.previousObservation = observation
        self.previousReward = -1.0
        return action

    def get_action(self, observation):
        self.totalMoves = self.totalMoves + 1
        if random.random() < self.totalMoves / 2000000:
            return self.max_arg(observation) # Exploitation
        else:
            return self.actionSpace.sample() # Exploration


    def max_lookup(self, observation):
        pos = observation[0]
        velocity = observation[1]
        maxR = self.Q.get((pos,velocity), 0)
        maxR = max(maxR, self.Q.get((pos, velocity, 0), 0))
        maxR = max(maxR, self.Q.get((pos, velocity, 1), 0))
        maxR = max(maxR, self.Q.get((pos, velocity, 2), 0))
        return maxR

    def max_arg(self, observation):
        pos = observation[0]
        velocity = observation[1]
        action = 0
        maxR = self.Q.get((pos, velocity, 0), 0)
        if maxR < self.Q.get((pos, velocity, 1), 0):
            maxR = self.Q.get((pos, velocity, 1), 0)
            action = 1
        if maxR < self.Q.get((pos, velocity, 2), 0):
            maxR = self.Q.get((pos, velocity, 2), 0)
            action = 2
        return action

