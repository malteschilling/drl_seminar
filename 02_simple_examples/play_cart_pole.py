import gym
env = gym.make('CartPole-v0')
#env = gym.make('MsPacman-v0')
for i_episode in range(1):
    observation = env.reset()
    for t in range(100):
        env.render()
        print(observation)
        action = env.action_space.sample()
        inp_key = raw_input()
        if (inp_key=='l'):
            print("Left")
            action = 0
        else:
            print("Right")
            action = 1
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break