import random

class QAgent():
    IniQ = 0
    actionSpace = None
    totalMoves = 0
    previousAction = None
    previousObservation = None
    previousReward = None
    alpha = 0.9
    gamma = 0.9
    Q = {} # table for Q-function
    frequency = {}
    def __init__(self, actionSpace):
        self.actionSpace = actionSpace

    def move(self, observation, reward, done):
        pos, velocity = self.process_observation(observation)
        if done:
            self.Q[(pos, velocity)] = reward
        pos, velocity = self.process_observation(self.previousObservation)
        s_a = (pos, velocity, self.previousAction)
        self.frequency[s_a] = self.frequency.get(s_a, self.IniQ) + 1
        this_q = self.Q.get(s_a, self.IniQ)
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
        pos, velocity = self.process_observation(observation)
        maxR = self.Q.get((pos,velocity),  self.IniQ)
        maxR = max(maxR, self.Q.get((pos, velocity, 0), self.IniQ))
        maxR = max(maxR, self.Q.get((pos, velocity, 1), self.IniQ))
        maxR = max(maxR, self.Q.get((pos, velocity, 2), self.IniQ))
        return maxR

    def max_arg(self, observation):
        pos, velocity = self.process_observation(observation)
        action = 0
        maxR = self.Q.get((pos, velocity, 0), self.IniQ)
        if maxR < self.Q.get((pos, velocity, 1), self.IniQ):
            maxR = self.Q.get((pos, velocity, 1), self.IniQ)
            action = 1
        if maxR < self.Q.get((pos, velocity, 2), self.IniQ):
            maxR = self.Q.get((pos, velocity, 2), self.IniQ)
            action = 2
        return action

    def process_observation(self, observation):
        pos = int(observation[0] * 100)
        velocity = int(observation[1] * 100)
        return pos, velocity