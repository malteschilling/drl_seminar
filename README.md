# drl_seminar
## Deep Reinforcement Learning Course 2017

Repository that includes basic python implementations of the discussed (deep) reinforcement learning algorithms. The seminar is held at Bielefeld University (winter semester 2017/18).

## Course Description
Many seemingly simple tasks are hard to describe in a formal way. Nonetheless, humans are quite good to solve such problems and manage such tasks. Crucial is the ability to learn which is a characteristic and prerequisite for intelligent behavior. Machine Learning in general aims at learning how to solve a task through training instead of relying on a formal description. 

Neural Networks provide a learning approach that addresses the problem of (very) high dimensional input data and how to project it into lower dimensional spaces. A neuronal network can therefore be understood as approximating a function in high dimensional spaces. Modern deep learning provides a very powerful framework for supervised learning. By adding more layers and more units within a layer, a deep network can represent functions of increasing complexity. Most tasks that consist of mapping an input vector to an output vector, and that are easy for a person to do rapidly, can be accom-plished via deep learning, given sufficiently large models and sufficiently large datasets of labeled training examples. Oth-er tasks, that can not be described as associating one vector to another, or that are diﬃcult enough that a person would require time to think and reflect in order to accomplish the task, remain beyond the scope of deep learning for now.
Reinforcement learning (RL) is usually about sequential decision making, solving problems in a wide range of fields in science, engineering and arts (Sutton and Barto, 2017). With recent exciting achievements of deep learning (LeCun et al., 2015; Goodfellow et al., 2016), benefiting from big data, powerful computation and new algorithmic techniques, we have been witnessing the renaissance of reinforcement learning (Krakovsky, 2016), especially, the combination of reinforce-ment learning and deep neural networks, i.e., deep reinforcement learning (deep RL).

The seminar focus is on current approaches to deep reinforcement learning as has been explored widely in the last cou-ple of years in the area of learning decisions in computer games. Further, current work extends these approaches to more real world problems as grasping in robotics. 
The seminar will give an introduction into the theoretical background of Neural Networks, Deep Learning and Reinforce-ment Learning. It will afterwards deal with state of the art methods and research literature presenting those methods. The seminar aims at comparing different approaches and providing an overview on current evolving principles and questions.

## Installation
The code is written in python (2.7) – and will reference the original source as well as the original publication.

Requirements:
* [keras](https://keras.io) (for deep learning)
* which either requires [tensorflow](https://www.tensorflow.org) or [theano](http://deeplearning.net/software/theano/)
* the [OpenAI gym](https://gym.openai.com/)

Installation:
I recommend to install all the requirements in a virtual environment or container. The easiest way is to follow the instructions on the given sites:
1. [Tensorflow](https://www.tensorflow.org/install/): Follow the installation guidelines for your plattform — again, recommended is choosing virtualenv (or docker). Afterwards make sure to activate that virtual environment.
2. [Keras](https://keras.io/#installation)
3. [OpenAI gym](https://gym.openai.com/docs/)

After installation, the [OpenAI webpages](https://gym.openai.com/docs/) provide basic information on how to get started and how to add further environments (the algorithms will be applied to the Atari games which should be installed).

## Usage
There will be example code (numbers indicate each week) that can be used, e.g. for the second week this only contains the cart pole balancing task and how this could be controlled through the keyboard (press 'l' and Return for movement to the left, any other key for movement to the right).
> python play\_cart\_pole.py
