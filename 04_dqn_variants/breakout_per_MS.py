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
from SumTree import SumTree

EPISODES = 20000

# Code is from https://github.com/rlcode/reinforcement-learning
# adapted to Breakout-RAM scenario
# and introduced the Prioritized Experience Replay (proportional)
# using a SumTree as a Memory structure

class PERAgent:
    def __init__(self, action_size):
        self.render = False
        self.load_model = False
        # environment settings
        self.state_size = (128,4)
        self.action_size = action_size
        # parameters about epsilon
        self.epsilon = 1.
        self.epsilon_start, self.epsilon_end = 1.0, 0.1
        self.exploration_steps = 1000000.
        self.epsilon_decay_step = (self.epsilon_start - self.epsilon_end) \
                                  / self.exploration_steps
        # parameters about training
        self.batch_size = 32
        self.learning_rate = 0.00025
        self.train_start_episode = 200 #50000
        self.update_target_rate = 10000
        self.discount_factor = 0.99
        #self.memory = deque(maxlen=400000)
        self.memory = SumTree(100000)
        self.no_op_steps = 30
        # build model
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_model()

        # Prioritized Experience Replay parameters. See https://arxiv.org/pdf/1511.05952.pdf
        #self.per_proportional_prioritization = per_proportional_prioritization  # Flavour of Prioritized Experience Rep.
        #self.per_apply_importance_sampling = per_apply_importance_sampling
        #self.prio_max = 0
        self.per_epsilon = 1E-6
        self.per_alpha = 1.0
        #self.per_beta0 = 0.4
        #self.per_beta = self.per_beta0

        self.optimizer = self.optimizer()

        self.sess = tf.InteractiveSession()
        K.set_session(self.sess)

        self.avg_q_max, self.avg_loss = 0, 0
        self.summary_placeholders, self.update_ops, self.summary_op = \
            self.setup_summary()
        self.summary_writer = tf.summary.FileWriter(
            'summary/breakout_per', self.sess.graph)
        self.sess.run(tf.global_variables_initializer())

        if self.load_model:
            self.model.load_weights("./save_model/breakout_per.h5")

    # if the error is in [-1, 1], then the cost is quadratic to the error
    # But outside the interval, the cost is linear to the error
    def optimizer(self):
        a = K.placeholder(shape=(None,), dtype='int32')
        y = K.placeholder(shape=(None,), dtype='float32')

        py_x = self.model.output

        a_one_hot = K.one_hot(a, self.action_size)
        q_value = K.sum(py_x * a_one_hot, axis=1)
        error = K.abs(y - q_value)

        quadratic_part = K.clip(error, 0.0, 1.0)
        linear_part = error - quadratic_part
        loss = K.mean(0.5 * K.square(quadratic_part) + linear_part)

        optimizer = RMSprop(lr=self.learning_rate, epsilon=0.01)
        updates = optimizer.get_updates(self.model.trainable_weights, [], loss)
        train = K.function([self.model.input, a, y], [loss], updates=updates)

        return train

    # approximate Q function using Convolution Neural Network
    # state is input and Q Value of each action is output of network
    def build_model(self):
        model = Sequential()
        #model.add(Conv2D(32, (8, 8), strides=(4, 4), activation='relu',
         #                input_shape=self.state_size))
        #model.add(Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))
        #model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
        #model.add(Flatten())
        model.add(Reshape((self.state_size[0]*self.state_size[1],), input_shape=self.state_size))
        model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(self.action_size))
        model.summary()
        return model

    # after some time interval update the target model to be same with model
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    # get action from model using epsilon-greedy policy
    def get_action(self, history):
        history = np.float32(history / 255.0)
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            q_value = self.model.predict(history)
            return np.argmax(q_value[0])

    def calculate_proportional_error(self, td_error):
        # Using the proportional error:
        # The error is first converted to priority using this formula:
        #     p = (error + \epsilon)^\alpha
        # Epsilon is a small positive constant that ensures that no transition has zero priority.
        # Alpha, 0 <= \alpha <= 1, controls the difference between high and low error. 
        # It determines how much prioritization is used. With \alpha = 0 we would get the uniform case.
        return np.power( (td_error + self.per_epsilon), self.per_alpha)

    # save sample <s,a,r,s'> to the replay memory
    def add_to_replay_memory(self, history, action, reward, next_history, dead, td_error):
        experience = (history, action, reward, next_history, dead)
        # https://jaromiru.com/2016/11/07/lets-make-a-dqn-double-learning-and-prioritized-experience-replay/
        # Using the proportional error:
        # The error is first converted to priority using this formula:
        #     p = (error + \epsilon)^\alpha
        # Epsilon is a small positive constant that ensures that no transition has zero priority.
        # Alpha, 0 <= \alpha <= 1, controls the difference between high and low error. 
        # It determines how much prioritization is used. With \alpha = 0 we would get the uniform case.
        self.memory.add( self.calculate_proportional_error(td_error) , experience)
        #self.memory.append((history, action, reward, next_history, dead))

    # pick samples randomly from replay memory (with batch_size)
    def train_replay(self):
        if self.epsilon > self.epsilon_end:
            self.epsilon -= self.epsilon_decay_step

        idx, priorities, mini_batch = self.memory.sample(self.batch_size)
        #mini_batch = random.sample(self.memory, self.batch_size)

        history = np.zeros((self.batch_size, self.state_size[0],
                            self.state_size[1]))
        next_history = np.zeros((self.batch_size, self.state_size[0],
                                 self.state_size[1]))
        target = np.zeros((self.batch_size,))
        action, reward, dead = [], [], []

        for i in range(self.batch_size):
            history[i] = np.float32(mini_batch[i][0] / 255.)
            next_history[i] = np.float32(mini_batch[i][3] / 255.)
            action.append(mini_batch[i][1])
            reward.append(mini_batch[i][2])
            dead.append(mini_batch[i][4])
        
        value = self.model.predict(next_history)
        target_value = self.target_model.predict(next_history)
        # TD Error calculation following http://blog.bsu.me/294-per-doom/
        max_value = np.argmax(value, axis=1)
        td_error = np.abs(np.choose(max_value, np.transpose(value)) - (reward + agent.discount_factor * np.choose(max_value, np.transpose(target_value))))
        priorities = self.calculate_proportional_error(td_error)
        # like Q Learning, get maximum Q value at s'
        # But from target model
        for i in range(self.batch_size):
            # Update priorities in replay memory
            self.memory.update(idx[i], priorities[i])
            if dead[i]:
                target[i] = reward[i]
            else:
                # the key point of Double DQN
                # selection of action is from model
                # update is from target model
                target[i] = reward[i] + self.discount_factor * \
                                        target_value[i][np.argmax(value[i])]
                                        #np.amax(target_value[i])

        loss = self.optimizer([history, action, target])
        self.avg_loss += loss[0]

    # make summary operators for tensorboard
    def setup_summary(self):
        episode_total_reward = tf.Variable(0.)
        episode_avg_max_q = tf.Variable(0.)
        episode_duration = tf.Variable(0.)
        episode_avg_loss = tf.Variable(0.)

        tf.summary.scalar('Total Reward/Episode', episode_total_reward)
        tf.summary.scalar('Average Max Q/Episode', episode_avg_max_q)
        tf.summary.scalar('Duration/Episode', episode_duration)
        tf.summary.scalar('Average Loss/Episode', episode_avg_loss)

        summary_vars = [episode_total_reward, episode_avg_max_q,
                        episode_duration, episode_avg_loss]
        summary_placeholders = [tf.placeholder(tf.float32) for _ in
                                range(len(summary_vars))]
        update_ops = [summary_vars[i].assign(summary_placeholders[i]) for i in
                      range(len(summary_vars))]
        summary_op = tf.summary.merge_all()
        return summary_placeholders, update_ops, summary_op


if __name__ == "__main__":
    # In case of BreakoutDeterministic-v3, always skip 4 frames
    # Deterministic-v4 version use 4 actions
    env = gym.make('Breakout-ram-v4')
    agent = PERAgent(action_size=3)

    scores, episodes, global_step = [], [], 0

    for e in range(EPISODES):
        done = False
        dead = False
        # 1 episode = 5 lives
        step, score, start_life = 0, 0, 5
        observe = env.reset()

        # this is one of DeepMind's idea.
        # just do nothing at the start of episode to avoid sub-optimal
        for _ in range(random.randint(1, agent.no_op_steps)):
            state, _, _, _ = env.step(1)

        # At start of episode, there is no preceding frame
        # So just copy initial states to make history
        history = np.stack((state, state, state, state), axis=1)
        history = np.reshape([history], (1, 128, 4))

        while not done:
            if agent.render:
                env.render()
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

            reward = np.clip(reward, -1., 1.)
            # TD Error calculation following http://blog.bsu.me/294-per-doom/
            value_p, = agent.model.predict(next_history)
            target_value_p, = agent.target_model.predict(next_history)
            td_error = np.abs(value_p[np.argmax(value_p)]/255. - (reward + agent.discount_factor * target_value_p[np.argmax(value_p)]/255.))
            agent.add_to_replay_memory(history, action, reward, next_history, dead, td_error)
            # every some time interval, train model
            if e > agent.train_start_episode: #len(self.memory) < self.train_start:
                agent.train_replay()
            # update the target model with model
            if global_step % agent.update_target_rate == 0:
                agent.update_target_model()

            score += reward

            # if agent is dead, then reset the history
            if dead:
                dead = False
            else:
                history = next_history

            # if done, plot the score over episodes
            if done:
                if global_step > agent.train_start_episode:
                    stats = [score, agent.avg_q_max / float(step), step,
                             agent.avg_loss / float(step)]
                    for i in range(len(stats)):
                        agent.sess.run(agent.update_ops[i], feed_dict={
                            agent.summary_placeholders[i]: float(stats[i])
                        })
                    summary_str = agent.sess.run(agent.summary_op)
                    agent.summary_writer.add_summary(summary_str, e + 1)

                print("episode:", e, "  score:", score, "  epsilon:", agent.epsilon,
                      "  global_step:", global_step, "  average_q:",
                      agent.avg_q_max / float(step), "  average loss:",
                      agent.avg_loss / float(step))

                agent.avg_q_max, agent.avg_loss = 0, 0

        if e % 1000 == 0:
            agent.model.save_weights("./save_model/breakout_per.h5")
