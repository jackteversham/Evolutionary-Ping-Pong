# Evolutionary-Ping-Pong

The code in this repository is used to train a reinforcement learning agent using an evolutionary strategy as apposed to traditional RL methods.

pong_estool.py and pong_estool_parallel.py train the agent in a sequential and parallel fashion respectively.

The policy of the agent takes the form of a neural network -- the weights of which are evolved over time out of a population of noisy candiate solutions. The best solutions 
perform well and are propagated into the population of the next generation, a survival of the fittest approach.

The environment used is OpenAI Gym and the ES framework can be found here https://github.com/hardmaru/estool.
A custom ES framework was also built -- please contact me for more information.
