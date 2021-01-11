#%%
import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt

#%%
#first create a data set
#generate a toy 2D regression dataset
sz = 100
X,Y = np.meshgrid(np.linspace(-1,1,sz),np.linspace(-1,1,sz))
mux,muy,sigma=0.3,-0.3,4
G1 = np.exp(-((X-mux)**2+(Y-muy)**2)/2.0*sigma**2)
mux,muy,sigma=-0.3,0.3,2
G2 = np.exp(-((X-mux)**2+(Y-muy)**2)/2.0*sigma**2)
mux,muy,sigma=0.6,0.6,2
G3 = np.exp(-((X-mux)**2+(Y-muy)**2)/2.0*sigma**2)
mux,muy,sigma=-0.4,-0.2,3
G4 = np.exp(-((X-mux)**2+(Y-muy)**2)/2.0*sigma**2)
G = G1 + G2 - G3 - G4
fig,ax = plt.subplots()
im = ax.imshow(G, vmin=-1, vmax=1, cmap='jet')
#plt.axis('off')

#%%
#begin with a population of random input parameters jittered with 
np.random.seed(5)
learningRate = 0.05 #define the learning rate
stdDev = 5
iterations = 7 #allow 5 iterations for convergence

populationSize = 401

noise = np.random.randn(populationSize, 2) #noise for 2D values
startPoint = np.array([70, 60.0]) #set a starting value, could be random



plt.figure(figsize=(20,5))
for i in range(iterations):
    
    startPointNoise = np.expand_dims(startPoint,0) + stdDev * noise; #startPoint with added noise
    
    
    ax1 = plt.subplot(1,iterations,i+1)
    plt.imshow(G, vmin=-1, vmax=1, cmap='jet')
    
    x,y = zip(*startPointNoise) #seprate [x,y] in to x and y
    plt.scatter(x,y,4,'k', edgecolors = "face")
    plt.show()
    

    rewards = np.array([G[int(element[1]), int(element[0])] for element in startPointNoise]) #each point evaluated
    rewards_indexes = np.argsort(rewards)
    
    rewards -= rewards.mean() #subtract the mean
    rewards /= rewards.std() #normalise to be in range (0,1)
   
    
    x_weighted_sum = 0
    for j in rewards_indexes:
        x_weighted_sum = x_weighted_sum + abs(rewards[j]*startPointNoise[j]) #weighted sum proportional to rewards
                                                                             #we want to propagate good guesses
    
    wsum = np.dot(rewards, noise) #weighted sum proportional to reward
    print(rewards)
    print(noise)
    print(wsum) 
   
    delta = wsum*learningRate
    
    startPoint += delta

plt.show()
#%%
print(startPointNoise.shape)

     

# %%

