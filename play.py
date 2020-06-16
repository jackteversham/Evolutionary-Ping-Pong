from keras.layers import Dense
from keras.models import Sequential

from datetime import datetime
import utils
import simple_es as es
import view_weights as wViewer
import numpy as np
import gym
import h5py
import os
import sys

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
    
    stdDeviation = np.zeros(401)
    delta = np.zeros(numWeights)
    weightConfigs, populationSize, noise = es.generateWeightPopulation(startPoint, numWeights, numGenerations, stdDeviation, CMA)
    
    print("Generating weights population jittered with noise...")
    
    bias0, kernel0, bias1, kernel1 = es.reshapeWeights(weightConfigs)
    createWeightsFiles(bias0, kernel0, bias1, kernel1, populationSize)
  
    #os.environ['KMP_DUPLICATE_LIB_OK']='True'
    if (os.path.isfile('james_model_weights.h5')):
        print("loading model weights...")
        model.load_weights('weightFiles/weights0.h5', by_name=True)
        
    else:
        print("Error: could not load previous model weights")
        sys.exit(0)
    # gym initialization
    env = gym.make("Pong-v0")
    observation = env.reset()
    prev_input = None

    # Define actions according to the Gym environment
    UP_ACTION = 2
    DOWN_ACTION = 3    

    # gamma value for discounted rewards
    gamma = 0.99
  
    episode_nb = 0 #tracks the episode that we are on. 1 episode = first player to 21 points
    reward_sum = 0

    action = np.random.randint(UP_ACTION, DOWN_ACTION+1) # create first action randomly to supply to the environment step call below
    #print(model.summary())
    while numGenerations != 10:
        rewards = np.zeros(populationSize) #reward sum per episode
        print("\n----------Generation "+str(numGenerations)+"----------")
        toLoad = 'weightFiles/weights'+str(episode_nb)+'.h5'
        model.load_weights(toLoad, by_name=True) #load the next set of weights
        print("Episode "+str(episode_nb)+". Model has loaded weights"+str(episode_nb)+".h5")

        while True:
            
            # choose random action
            #env.render() #show the environment
            #time.sleep(0.02)
            observation, reward, done, info = env.step(action)
            #print(info)
            if reward > 0:
                reward += 1
                print(reward)
            reward_sum += reward #add reward of previous action to the sum
           
            skip_frame = False
            cur_input = utils.preprocess(observation)

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
            
            # reset environment at the end of an episode (first player to 21 points)
            if done:
                observation = env.reset()
                rewards[episode_nb] = reward_sum
                reward_sum = 0
                
                episode_nb+=1
                if episode_nb == populationSize:
                    episode_nb=0
                    break

                toLoad = 'weightFiles/weights'+str(episode_nb)+'.h5'
                model.load_weights(toLoad, by_name=True) #load the next set of weights
                print("Episode "+str(episode_nb)+". Model has loaded weights"+str(episode_nb)+".h5")
                action = np.random.randint(UP_ACTION, DOWN_ACTION+1) # create first action for next episode

        CMA = True
        numGenerations +=1
            #delta = es.evolve(weightConfigs, rewards, noise) 
        startPoint, stdDeviation = es.CMAesEvolve(weightConfigs, rewards, populationSize)
        #print(startPoint)
        #print("NEW START POINT")
        weightConfigs, populationSize, noise = es.generateWeightPopulation(startPoint, numWeights, numGenerations, stdDeviation, CMA)
        print("\nGenerating weights population jittered with noise...")
         
        bias0, kernel0, bias1, kernel1 = es.reshapeWeights(weightConfigs)
        createWeightsFiles(bias0, kernel0, bias1, kernel1, populationSize)
        

        

def createWeightsFiles(bias0, kernel0, bias1, kernel1, populationSize):

    for i in range(populationSize):
        fileName  = "weightFiles/weights"+str(i)+'.h5'
        f = h5py.File(fileName,"w") #we have to write to create
        f.create_dataset('dense_3/dense_3/bias:0', data = bias0[i])
        f.create_dataset('dense_3/dense_3/kernel:0', data = kernel0[i].reshape((6,50)))
        f.create_dataset('dense_4/dense_4/bias:0', data = bias1[i])
        f.create_dataset('dense_4/dense_4/kernel:0', data = kernel1[i])
        f.close()
       # f = h5py.File(fileName,"a") #now we want to append


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