import gym
import random
import numpy as np
import tensorflow as tf
from collections import deque
from skimage.color import rgb2gray
from skimage.transform import resize
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.layers import Dense, Reshape, Flatten
from keras.layers.convolutional import Conv2D
from keras import backend as K

EPISODES = 50000


class DQNAgent:
    def __init__(self, action_size):
        self.render = True
        self.load_model = True
        # environment settings
        self.state_size = (128,4)
        self.action_size = action_size

        # build model
        self.model = self.build_model()

        self.sess = tf.InteractiveSession()
        K.set_session(self.sess)

        self.avg_q_max, self.avg_loss = 0, 0
        self.sess.run(tf.global_variables_initializer())

        if self.load_model:
            self.model.load_weights("./save_model/breakout_per_trained.h5")

    # approximate Q function using Convolution Neural Network
    # state is input and Q Value of each action is output of network
    def build_model(self):
        model = Sequential()
        model.add(Reshape((self.state_size[0]*self.state_size[1],), input_shape=self.state_size))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(self.action_size))
        model.summary()
        return model

    # get action from model using epsilon-greedy policy
    def get_action(self, history):
        history = np.float32(history / 255.0)
        q_value = self.model.predict(history)
        return np.argmax(q_value[0])

from gym.envs.classic_control import rendering
def repeat_upsample(rgb_array, k=1, l=1, err=[]):
    # repeat kinda crashes if k/l are zero
    if k <= 0 or l <= 0: 
        if not err: 
            print "Number of repeats must be larger than 0, k: {}, l: {}, returning default array!".format(k, l)
            err.append('logged')
        return rgb_array

    # repeat the pixels k times along the y axis and l times along the x axis
    # if the input image is of shape (m,n,3), the output image will be of shape (k*m, l*n, 3)

    return np.repeat(np.repeat(rgb_array, k, axis=0), l, axis=1)

if __name__ == "__main__":
    viewer = rendering.SimpleImageViewer()
    # In case of BreakoutDeterministic-v3, always skip 4 frames
    # Deterministic-v4 version use 4 actions
    env = gym.make('Breakout-ram-v4')
    agent = DQNAgent(action_size=3)

    scores, episodes, global_step = [], [], 0
    summed_score = 0.
    for e in range(EPISODES):
        done = False
        dead = False
        # 1 episode = 5 lives
        step, score, start_life = 0, 0, 5
        observe = env.reset()

        # this is one of DeepMind's idea.
        # just do nothing at the start of episode to avoid sub-optimal
        #for _ in range(random.randint(1, agent.no_op_steps)):
        state, _, _, _ = env.step(1)
        # At start of episode, there is no preceding frame
        # So just copy initial states to make history
        history = np.stack((state, state, state, state), axis=1)
        history = np.reshape([history], (1, 128, 4))

        while not done:
            if agent.render:
                rgb = env.render('rgb_array')
                upscaled=repeat_upsample(rgb,3, 3)
                viewer.imshow(upscaled)

            global_step += 1
            step += 1

            # get action for the current history and go one step in environment
            action = agent.get_action(history)
            # change action to real_action
            if action == 0:
                real_action = 1
            elif action == 1:
                real_action = 2
            else:
                real_action = 3

            next_state, reward, done, info = env.step(real_action)
            next_state = np.reshape([next_state], (1, 128, 1))
            next_history = np.append(next_state, history[:, :, :3], axis=2)
            agent.avg_q_max += np.amax(
                agent.model.predict(np.float32(history / 255.))[0])

            # if the agent missed ball, agent is dead --> episode is not over
            if start_life > info['ale.lives']:
                dead = True
                start_life = info['ale.lives']

            score += reward

            # if agent is dead, then reset the history
            if dead:
                dead = False
            else:
                history = next_history

            # if done, plot the score over episodes
            if done:
                summed_score += score
                print("episode:", e, "  score:", score, "  average_q:",
                      agent.avg_q_max / float(step), " - mean score:", (summed_score/(e+1)) )

                agent.avg_q_max, agent.avg_loss = 0, 0