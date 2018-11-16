import numpy as np
import gym

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy, EpsGreedyQPolicy, LinearAnnealedPolicy
from rl.memory import SequentialMemory


ENV_NAME = 'MountainCar-v0'

# Get the environment and extract the number of actions.
env = gym.make(ENV_NAME)
np.random.seed(12345)
env.seed(12345)
nb_actions = env.action_space.n

# Next, we build a very simple model.

model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(64, kernel_initializer='random_uniform',
                bias_initializer='zero'))
model.add(Activation('relu'))
# model.add(Dropout(rate = 0.1))
# model.add(Dense(26, kernel_initializer='random_uniform', bias_initializer='zeros'))
# model.add(Activation('relu'))
# model.add(Dense(16))
# model.add(Activation('relu'))
model.add(Dense(nb_actions, kernel_initializer='random_uniform'))
model.add(Activation('linear'))
print(model.summary())

memory = SequentialMemory(limit=50000, window_length=1)
policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr = 'eps', value_max=.5, value_min=.0, value_test=.05, nb_steps=400000)
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=100,
               target_model_update=1e-4, policy=policy, gamma= 0.99)
dqn.compile(Adam(), metrics=['mae'])
dqn.load_weights('10step.h5f'.format(ENV_NAME))
dqn.fit(env, nb_steps=400000, visualize=False, verbose=2, action_repetition= 5)

# After training is done, we save the final weights.
dqn.save_weights('dqn_{}_weights.h5f'.format(ENV_NAME), overwrite=True)

# Finally, evaluate our algorithm for 5 episodes.
dqn.test(env, nb_episodes=5, visualize=True)