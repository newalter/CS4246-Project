import numpy as np
import gym

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory




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



# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and

# even the metrics!

memory = SequentialMemory(limit=50000, window_length=1)

policy = EpsGreedyQPolicy()

dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,

               target_model_update=1e-4, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

dqn.load_weights('dqn_{}_weights.h5f'.format(ENV_NAME))

dqn.test(env, nb_episodes=5, visualize=True)