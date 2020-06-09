#%%
import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt

def generateWeightPopulation(startPoint, numWeights):
    #begin with a population of random input parameters jittered with 
    np.random.seed(5)
    stdDev = 10 #standard deviation of noised attached to weights guesses

    populationSize = 20 #number of noisy weight combinations to try
   
    weightConfigs = np.zeros((populationSize,numWeights)) #this is our population of guesses for the weights of the NN
    noisePopulation = np.zeros((populationSize, numWeights))

    for i in range(populationSize): 
        noise = np.random.randn(numWeights) #guassian noise for guessed weight values
        noisePopulation[i] = noise
        weightConfigs[i] = startPoint + stdDev * noise; #jitter our startPoint with gaussian noise

    return weightConfigs, populationSize, noisePopulation


def reshapeWeights(weightConfigs): #reshape input weight array such that an appropriate .hdf5 weight file can be created

    bias0 = weightConfigs[:, 0:50] #bias of input layer
    kernel0 = weightConfigs[:, 50:350] #6 input features with hidden layer of 50 neurons
    bias1 = weightConfigs[:,350:351 ] #bias of output weights
    kernel1 = weightConfigs[:,351:401 ] #output weight kernel 

    return bias0, kernel0, bias1, kernel1

def evolve(weightConfigs, rewards, noise):
    learningRate = 0.08 #define the learning rate

    print(rewards.std())
    if rewards.std() !=0:
        rewards -= rewards.mean()
        rewards /= rewards.std()
    

    weightedSum = np.dot(rewards, noise) #weighted sum proportional to reward
    delta = weightedSum*learningRate
    print(delta.shape)
    
    return delta

def CMAesEvolve(weightConfigs, rewards, populationSize):

    bestIndices = np.argsort(rewards)#sort based on fitness, we get indexes out

    numberOfBestSolutions = (bestIndices.size)*0.25 #we want the best 25% of the previous generation
    print(numberOfBestSolutions)
    mean = np.mean(weightConfigs, axis=0) #a column wise mean 
    #covariance = np.cov()
    print(mean.shape)
    print(mean)

    









# print(weightConfigs[0][0])
# print(weightConfigs[3][0])



#%%

# plt.figure(figsize=(20,5))
# for i in range(iterations):
    
    
    
    
#     ax1 = plt.subplot(1,iterations,i+1)
#     plt.imshow(G, vmin=-1, vmax=1, cmap='jet')
    
#     x,y = zip(*startPointNoise) #seprate [x,y] in to x and y
#     plt.scatter(x,y,4,'k', edgecolors = "face")
#     plt.show()
    

#     rewards = np.array([G[int(element[1]), int(element[0])] for element in startPointNoise]) #each point evaluated
#     rewards_indexes = np.argsort(rewards)
    
#     rewards -= rewards.mean() #subtract the mean
#     rewards /= rewards.std() #normalise to be in range (0,1)
   
    
#     x_weighted_sum = 0
#     for j in rewards_indexes:
#         x_weighted_sum = x_weighted_sum + abs(rewards[j]*startPointNoise[j]) #weighted sum proportional to rewards
#                                                                              #we want to propagate good guesses
    
#     wsum = np.dot(rewards, noise) #weighted sum proportional to reward
   
#     delta = wsum*learningRate
    
#     startPoint += delta

# plt.show()
# #%%
# print(startPointNoise.shape)

     

# # %%
