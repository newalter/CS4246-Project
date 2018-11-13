import gym

from Agent import QAgent

EPISODES = 10000

env = gym.make('MountainCar-v0')
agent = QAgent(env.action_space)

AvgReward = 0

for episode in range(EPISODES):
    totalReward = 0
    observation = env.reset()
    done = False
    action = agent.new_episode(observation= observation)
    while not done:
        observation, reward, done, info = env.step(action)
        # env.render()
        totalReward += reward
        action = agent.move(observation, reward, done)
        # print(observation, '  ', action)
    print(totalReward)
    AvgReward += totalReward

print(AvgReward / EPISODES)
env.close()