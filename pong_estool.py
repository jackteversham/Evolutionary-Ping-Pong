from keras.layers import Dense
from keras.models import Sequential

from datetime import datetime
import utils
from estool import es
import view_weights as wViewer
import numpy as np
import gym
import h5py
import os
import sys
import threading

def construct_model():
    FEATURE_SIZE = 6
    # creates a generic neural network architecture
    model = Sequential()
    # ReLU activation to allow network to represent non-linear relationships
    model.add(Dense(units=50,input_dim=FEATURE_SIZE, activation='relu', kernel_initializer='glorot_uniform', name = 'dense_3'))
    # output layer
    model.add(Dense(units=1, activation='sigmoid', kernel_initializer='RandomNormal', name = 'dense_4'))
    # use binary cross entropy loss function with Adam
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model    

def main():
    numGenerations = 0
    model = construct_model()
    print("\n\nModel Constructed.")
    numWeights = 401 #number of weights in the Neural network

    customStartPoint = True
    CMA = False
    
    startPoint = np.random.randn(numWeights) #random values form normal distribution

    if customStartPoint:
        startPoint = wViewer.getStartPoint()
        print("Custom Start Point Loaded.")
    
    # stdDeviation = np.zeros(401)
    # delta = np.zeros(numWeights)
    # weightConfigs, populationSize, noise = es.generateWeightPopulation(startPoint, numWeights, numGenerations, stdDeviation, CMA)
    
    # print("Generating weights population jittered with noise...")
    
    # bias0, kernel0, bias1, kernel1 = es.reshapeWeights(weightConfigs)
    # createWeightsFiles(bias0, kernel0, bias1, kernel1, populationSize)
  
    #os.environ['KMP_DUPLICATE_LIB_OK']='True'
    if (os.path.isfile('james_model_weights.h5')):
        print("loading model weights...")
        model.load_weights('weightFiles/weights0.h5', by_name=True)
        
    else:
        print("Error: could not load previous model weights")
        sys.exit(0)
    # gym initialization
    env = gym.make("Pong-v0") #make the gym environment
    observation = env.reset()
    prev_input = None

    # Define actions according to the Gym environment
    UP_ACTION = 2
    DOWN_ACTION = 3    

    # gamma value for discounted rewards
    gamma = 0.99
  
    episode_nb = 0 #tracks the episode that we are on. 1 episode = first player to 21 points
    reward_sum = 0
    count = 0

    action = np.random.randint(UP_ACTION, DOWN_ACTION+1) # create first action randomly to supply to the environment step call below
    #print(model.summary())
    MY_REQUIRED_FITNESS = 0
    NUM_PARAMETERS = 401
    POP_SIZE = 100
    INIT_STDDEV = 0.1
    WEIGHT_DECAY = 0.0

    solver = es.CMAES(NUM_PARAMETERS, INIT_STDDEV, POP_SIZE, WEIGHT_DECAY) #initiate a CMA evolution strategy (NUM_PARAMETERS)
    
    while True:
        count+=1
        print("-----Generation "+str(count) +"-----")
        
        solutions = solver.ask() #ask our solver for candidate solutions
        fitnessList = np.zeros(solver.popsize) #array to hold the fitness of our solutions

        for i in range(solver.popsize): #for each solution

            print(i)
           #agent = Agent(solutions[i]) #give the agent a solution
            bias0, kernel0, bias1, kernel1 = reshapeWeights(solutions[i])
            createWeightsFiles(bias0, kernel0, bias1, kernel1, solver.popsize, i)
            filename = 'weightFiles/weights' +str(i)+'.h5'
            model.load_weights(filename, by_name=True)
            fitnessList[i] = rollout(env, model, UP_ACTION, DOWN_ACTION)


        solver.tell(fitnessList)
        result = solver.result() #first element is the best solution, 2nd element is the best fitness
        
        print("Best fitness: ", result[1])


        if result[1] > MY_REQUIRED_FITNESS:
            break

            # choose random action
           
            #time.sleep(0.02)
        # observation, reward, done, info = env.step(action)
        #     #print(info)
        # if reward > 0:
        #     reward += 1
        #     print(reward)
        # reward_sum += reward #add reward of previous action to the sum
           
        # skip_frame = False
        # cur_input = utils.preprocess(observation)

        # if not utils.check_frame(cur_input):
        #     skip_frame = True
        #         #line below will ensure that on reset, at least one frame is passed to get a valid prev_input so that relative values can be calc from second frame
        #     prev_input = None 
            
        # if not skip_frame: 
        #     if prev_input is not None:
        #         x = utils.extract_intertial_feat(cur_input, prev_input)
        #             # scale feature vector to align with scale expected from training
        #         x = utils.feat_scale(x, cur_input.shape)
        #             # forward the policy network and sample an action from the policy distribution
        #         proba = model.predict(np.expand_dims(x, axis=1).T)
        #         action = UP_ACTION if proba > 0.5 else DOWN_ACTION
                    
        #     prev_input = cur_input 
        # else:
        #     action = np.random.randint(UP_ACTION, DOWN_ACTION+1)
            
        #     # reset environment at the end of an episode (first player to 21 points)
        # if done:
        #     observation = env.reset()
        #     rewards[episode_nb] = reward_sum
        #     reward_sum = 0
        #     episode_nb+=1

        #     if episode_nb == populationSize:
        #         episode_nb=0
        #         break

def rollout(env, model, UP_ACTION, DOWN_ACTION):
    prev_input = None
    obs = env.reset()
    done = False
    total_reward = 0
    while not done:
        #env.render() #show the environment
        #a = agent.get_action(obs)

        skip_frame = False
        cur_input = utils.preprocess(obs)

        if not utils.check_frame(cur_input):
            skip_frame = True
                #line below will ensure that on reset, at least one frame is passed to get a valid prev_input so that relative values can be calc from second frame
            prev_input = None 
            
        if not skip_frame: 
            if prev_input is not None:
                x = utils.extract_intertial_feat(cur_input, prev_input)
                    # scale feature vector to align with scale expected from training
                x = utils.feat_scale(x, cur_input.shape)
                    # forward the policy network and sample an action from the policy distribution
                proba = model.predict(np.expand_dims(x, axis=1).T)
                action = UP_ACTION if proba > 0.5 else DOWN_ACTION
                    
            prev_input = cur_input 
        else:
            action = np.random.randint(UP_ACTION, DOWN_ACTION+1)


        obs, reward, done, info = env.step(action) #take the given action
        total_reward += reward #sum up the total rewards achieved by the agent
        
    if (total_reward > -21.0):
        print(total_reward)
    return total_reward
    
def reshapeWeights(weightConfigs): #reshape input weight array such that an appropriate .hdf5 weight file can be created

    bias0 = weightConfigs[0:50] #bias of input layer
    kernel0 = weightConfigs[50:350] #6 input features with hidden layer of 50 neurons
    bias1 = weightConfigs[350:351 ] #bias of output weights
    kernel1 = weightConfigs[351:401 ] #output weight kernel 

    return bias0, kernel0, bias1, kernel1

        

def createWeightsFiles(bias0, kernel0, bias1, kernel1, populationSize, i):

    #for i in range(populationSize):
    fileName  = "weightFiles/weights"+str(i)+'.h5'
    f = h5py.File(fileName,"w") #we have to write to create
    f.create_dataset('dense_3/dense_3/bias:0', data = bias0)
    f.create_dataset('dense_3/dense_3/kernel:0', data = kernel0)
    f.create_dataset('dense_4/dense_4/bias:0', data = bias1)
    f.create_dataset('dense_4/dense_4/kernel:0', data = kernel1)
    f.close()
       # f = h5py.File(fileName,"a") #now we want to append

#my subclass of threading class
class myThread(threading.Thread):
    def __init__(self, threadID, range, model): #takes an ID, the range of operation, the model as parameters
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.range = range
        self.model = model
    def run(self):
        


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        # graceful exit
        print('\n\nGame exiting.')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)