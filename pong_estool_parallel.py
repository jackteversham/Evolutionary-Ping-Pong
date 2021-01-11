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
import multiprocessing

def construct_model():
    FEATURE_SIZE = 6
    # creates a generic neural network architecture
    model = Sequential()
    # ReLU activation to allow network to represent non-linear relationships
    model.add(Dense(units=25,input_dim=FEATURE_SIZE, activation='tanh', kernel_initializer='glorot_uniform', name = 'dense_3'))
    # output layer
    model.add(Dense(units=1, activation='sigmoid', kernel_initializer='RandomNormal', name = 'dense_4'))
    # use binary cross entropy loss function with Adam
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model    


   
# def parallelRollout(threadID, weightRange, model, env, UP_ACTION, DOWN_ACTION, threadLock, fitnessList):
#         diff = weightRange[1]-weightRange[0] #diff is a proportion of the population size
#         for i in range(diff):
#             filename = 'weightFiles/weights' +str(weightRange[0]+i)+'.h5'
#             print("Thread "+str(threadID)+" loading "+filename)
#             #self.threadLock.acquire()
#             # model.load_weights(filename, by_name=True)
#             # fitnessList[weightRange[0]+i] = rollout(env, model, UP_ACTION, DOWN_ACTION)

   
def rollout(env, UP_ACTION, DOWN_ACTION, weightRange, threadID, fitnessList):
    model = construct_model() #without this, it hangs if we try pass in a model from main thread, dont know why
    diff = weightRange[1]-weightRange[0] #diff is a portion of the population 
    for i in range(diff):
        filename = 'weightFiles/weights' +str(weightRange[0]+i)+'.h5'
        model.load_weights(filename, by_name=True)
        print("Thread "+str(threadID)+" loading "+filename)

        action = np.random.randint(UP_ACTION, DOWN_ACTION+1)
        #print(model.summary())
        prev_input = None
        obs = env.reset()
        done = False
        total_reward = 0
        while not done:
            #env.render() #show the environment
            #a = agent.get_action(obs)
            #print("here "+str(threadID))
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
            #print("here "+str(reward))
        
        

        fitnessList[weightRange[0]+i] = total_reward        
        if (total_reward > -21.0):
            print(total_reward)

        if (total_reward > -15.0):
            filename = "goodWeights/saved_weights"+str(total_reward)+".h5"
            model.save_weights(filename)
            print("Saved model to disk.")
    return
        
    #return total_reward
    
def reshapeWeights(weightConfigs): #reshape input weight array such that an appropriate .hdf5 weight file can be created
    bias0 = weightConfigs[0:25] #bias of input layer
    kernel0 = weightConfigs[25:175] #6 input features with hidden layer of 10 neurons
    bias1 = weightConfigs[175:176 ] #bias of output weights
    kernel1 = weightConfigs[176:201] #output weight kernel 
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
      
#my subclass of threading class
# class myThread(threading.Thread):
#     def __init__(self, threadID, weightRange, model, env, UP_ACTION, DOWN_ACTION, threadLock, fitnessList): #takes an ID, the range of operation, the model as parameters
#         threading.Thread.__init__(self)
#         self.threadID = threadID
#         self.weightRange = weightRange #an array containing start and end points
#         self.model = model
#         self.env = env
#         self.UP_ACTION = UP_ACTION
#         self.DOWN_ACTION = DOWN_ACTION
#         self.threadLock = threadLock
#         self.fitnessList = fitnessList

#     def run(self):
#         diff = self.weightRange[1]-self.weightRange[0] #diff is a proportion of the population size
        
#         for i in range(diff):
#             filename = 'weightFiles/weights' +str(self.weightRange[0]+i)+'.h5'
#             print("Thread "+str(self.threadID)+" loading "+filename)
#             #self.threadLock.acquire()
#             self.model.load_weights(filename, by_name=True)
#             self.fitnessList[self.weightRange[0]+i] = rollout(self.env, self.model, self.UP_ACTION, self.DOWN_ACTION)
#             #self.threadLock.release()
    
#         #return result
    
if __name__ == "__main__":
    try: 
        manager = multiprocessing.Manager()
        
        # model2 = construct_model() #for thread 2
        # model3 = construct_model()
        # model4 = construct_model()
       
        print("\n\nModels Constructed.")
    
        env = gym.make("Pong-v0") #make the gym environment
        env2 = gym.make("Pong-v0") #make another environment for thread 2 to play in
        env3 = gym.make("Pong-v0") #make another environment for thread 3 to play in
        env4 = gym.make("Pong-v0") #make another environment for thread 4 to play in
        env5 = gym.make("Pong-v0") #make the gym environment
        env6 = gym.make("Pong-v0") #make another environment for thread 2 to play in
        env7 = gym.make("Pong-v0") #make another environment for thread 3 to play in
        env8 = gym.make("Pong-v0") #make another environment for thread 4 to play in

        # Define actions according to the Gym environment
        UP_ACTION = 2
        DOWN_ACTION = 3
        count = 0    

        NUM_THREADS = 2
        threadLock = threading.Lock()
    
        #print(model.summary())
        MY_REQUIRED_FITNESS = 0
        NUM_PARAMETERS = 201
        POP_SIZE = 80
        INIT_STDDEV = 0.5
        WEIGHT_DECAY = 0.0

        solver = es.CMAES(NUM_PARAMETERS, INIT_STDDEV, POP_SIZE, WEIGHT_DECAY) #initiate a CMA evolution strategy (NUM_PARAMETERS)
        outfile = open("output.txt", 'w')
        outfile.write("") #clear output file
        outfile.close()
        while True:
            outfile = open("output.txt", 'a')
            count+=1
            print("-----Generation "+str(count) +"-----")
            outfile.write("-----Generation "+str(count) +"-----\n")
            outfile.write(str(datetime.now())+"\n")
            
            solutions = solver.ask() #ask our solver for candidate solutions
            fitnessList = manager.list() #shared list to hold the fitness of our solutions
        
            for i in range(solver.popsize): #for each solution
                fitnessList.append(-21) #the worst score possible on the game

            #agent = Agent(solutions[i]) #give the agent a solution
                bias0, kernel0, bias1, kernel1 = reshapeWeights(solutions[i])
                createWeightsFiles(bias0, kernel0, bias1, kernel1, solver.popsize, i)
                #filename = 'weightFiles/weights' +str(i)+'.h5'
            #model.load_weights(filename, by_name=True)
            #fitnessList[i] = rollout(env, model, UP_ACTION, DOWN_ACTION)

            range1 = np.array([0,10 ])
            range2 = np.array([10, 20])
            range3 = np.array([20, 30])
            range4 = np.array([30, 40])
            range5 = np.array([40,50 ])
            range6 = np.array([50, 60])
            range7 = np.array([60, 70])
            range8 = np.array([70, 80])

            # fitnessList1 = fitnessList[0:5]
            # fitnessList2 = fitnessList[5:10]
            # fitnessList3 = fitnessList[10:15]
            # fitnessList4 = fitnessList[15:20]

            lock = multiprocessing.Lock()
            processes = []
            p1 = multiprocessing.Process(target=rollout, args=(env,UP_ACTION,DOWN_ACTION, range1, 1, fitnessList))
            processes.append(p1)
            p2 = multiprocessing.Process(target=rollout, args=(env2, UP_ACTION,DOWN_ACTION, range2, 2, fitnessList))
            processes.append(p2)
            p3 = multiprocessing.Process(target=rollout, args=(env3, UP_ACTION,DOWN_ACTION, range3, 3, fitnessList))
            processes.append(p3)
            p4 = multiprocessing.Process(target=rollout, args=(env4, UP_ACTION,DOWN_ACTION, range4, 4, fitnessList))
            processes.append(p4)
            p5 = multiprocessing.Process(target=rollout, args=(env5,UP_ACTION,DOWN_ACTION, range5, 5, fitnessList))
            processes.append(p5)
            p6 = multiprocessing.Process(target=rollout, args=(env6, UP_ACTION,DOWN_ACTION, range6, 6, fitnessList))
            processes.append(p6)
            p7 = multiprocessing.Process(target=rollout, args=(env7, UP_ACTION,DOWN_ACTION, range7, 7, fitnessList))
            processes.append(p7)
            p8 = multiprocessing.Process(target=rollout, args=(env8, UP_ACTION,DOWN_ACTION, range8, 8, fitnessList))
            processes.append(p8)

            for p in processes:
                p.start()

            for p in processes:
                p.join()
            fitnessList = np.array(fitnessList)

            print("Fitness list ",fitnessList)

            solver.tell(fitnessList)
            result = solver.result() #first element is the best solution, 2nd element is the best fitness
            
            print("Best fitness: ", result[1])
            outfile.write("Best fitness: " + str(result[1])+"\n")
            res = np.array(result[0])
            outfile.write(str(res.mean())+"\n")

            outfile.close()
            if result[1] > MY_REQUIRED_FITNESS:
                outfile = open("output.txt", 'a')
                outfile.write("\nModel saved to 'jacks_model.h5'")
                outfile.close()
               
                break

    except KeyboardInterrupt:
        # graceful exit
        print('\n\nGame exiting.')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)