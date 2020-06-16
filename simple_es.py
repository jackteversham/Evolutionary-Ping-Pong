#%%
import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt

def generateWeightPopulation(startPoint, numWeights, numGenerations, stdDeviation, CMA):
    #begin with a population of random input parameters jittered with 
    #np.random.seed(5)
    stdDev = 3 #standard deviation of noised attached to weights guesses
    learningRate = 0.3

    populationSize = 100 #number of noisy weight combinations to try
    if(numGenerations>0):
        populationSize = 100
   
    weightConfigs = np.zeros((populationSize,numWeights)) #this is our population of guesses for the weights of the NN
    noisePopulation = np.zeros((populationSize, numWeights))

    for i in range(populationSize): 
        noise = np.random.randn(numWeights) #guassian noise for guessed weight values
        noisePopulation[i] = noise
        if CMA:
            weightConfigs[i] = startPoint + learningRate * stdDeviation * noise
        
        else:
            weightConfigs[i] = startPoint + stdDev * noise; #jitter our startPoint with gaussian noise
            print("Here")

    print(weightConfigs[i])
    print("Weight Configuration")
        

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
    
    rewards -= rewards.mean()
    if rewards.std() !=0:
        rewards /= rewards.std()
        print(rewards)
        print("REWARDS")
        weightedSum = np.dot(rewards, noise) #weighted sum proportional to reward
        print(weightedSum)
        print("WEIGHTED SUM")
    
   
        delta = weightedSum*learningRate
    else:
        delta = np.random.randn(401) #inject more noise if no rewards received during previous generation
    
    print(delta)
    print("DELTA")
    return delta

def CMAesEvolve(weightConfigs, rewards, populationSize):

    bestIndices = np.argsort(rewards)#sort based on fitness, we get indexes out
    bestIndices = np.flip(bestIndices) #we want to get the index of the highest reward first
    print(rewards)
    print(bestIndices)
    currentMean = np.mean(weightConfigs, axis=0) #a column wise mean 

    
    numberOfBestSolutions = int((populationSize)*0.1) #we want the best 25% of the previous generation
    # for i in range(401):
    #     for j in range(numberOfBestSolutions):
    #         mean[i] += weightConfigs[bestIndices[j]][i]

    # mean /= numberOfBestSolutions #mean for the next generation (i.e. the startPoint)

    best25 = np.zeros((numberOfBestSolutions, 401))
    for i in range(numberOfBestSolutions):
        best25[i] = weightConfigs[bestIndices[i]] #from the argsorted rewards, find our best performing weight configurations

    mean = np.mean(best25, axis=0) #start point for the next generation
 
    
    #best25 -= currentMean #subtract the mean

    covariance = np.cov((best25).T) #best25 will be of shape (numberOfBestSolutions, 401)
    diagonal = np.diag(covariance)
    #diagonalAbs = np.absolute(diagonal)

   # signs = diagonal/diagonalAbs #get the sign (pos or neg) of the variance value

    stdDeviation = np.sqrt(diagonal)
    #stdDeviation *= signs # maintain the sign


    #covariance = np.sum(best25**2, axis=0)/numberOfBestSolutions

    print("\n\n")

    print(mean.shape)
    print(best25.shape)
   
    print(stdDeviation.shape) 
    print("Diagonal\n")
    print(diagonal)
    #print(signs)
    print(stdDeviation)

    return mean, stdDeviation
    
    

    #return the mean (the startPoint for the next generation), as well as the


    









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
