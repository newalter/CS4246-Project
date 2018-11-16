import numpy as np
import gym

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.agents.cem import CEMAgent
from rl.agents.sarsa import SARSAAgent
from rl.policy import BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory, EpisodeParameterMemory

ENV_NAME = 'MountainCar-v0'

# Get the environment and extract the number of actions.
env = gym.make(ENV_NAME)
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n

# Next, we build a very simple model.

model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))
print(model.summary())

memory = SequentialMemory(limit=50000, window_length=1)
policy = EpsGreedyQPolicy()

sarsa = SARSAAgent(model, nb_actions, policy= policy, test_policy=None, gamma=0.99, nb_steps_warmup=10, train_interval=1, delta_clip=np.inf)
sarsa.compile(Adam(lr=1e-3))

sarsa.fit(env, nb_steps=500000, visualize=False, verbose=2)

# After training is done, we save the final weights.
sarsa.save_weights('sarsa_{}_weights.h5f'.format(ENV_NAME), overwrite=True)

# Finally, evaluate our algorithm for 5 episodes.
sarsa.test(env, nb_episodes=5, visualize=True)
