import h5py
import numpy as np


f = h5py.File('weightFiles/weights0.h5', 'r')
f2 =  h5py.File('james_model_weights.h5', 'r')
#f = h5py.File('james_model_weights.h5', 'r')
print(list(f.attrs))

print(list(f.keys())) #h5 file acts like a dictionary => this gives the keys
grp1 = f['dense_4']
print(list(grp1.keys()))

# will get a list of layer names which you can use as index
grps = f['dense_4/dense_4/']
print(list(grps.keys()))
f.close()
f2.close()



def getStartPoint():
    f = h5py.File('goodBatchWeightFiles/weights10.h5', 'r')
    bias0 = np.array(f['dense_3/dense_3/bias:0'])
    kernel0 = np.array(f['dense_3/dense_3/kernel:0'])
    bias1 = np.array(f['dense_4/dense_4/bias:0'])
    kernel1 = np.array(f['dense_4/dense_4/kernel:0'])

    dataset = np.concatenate([bias0.flatten(),kernel0.flatten(), bias1.flatten(), kernel1.flatten()])
    print(dataset.shape)
    f.close()
    return dataset

#print(dset.name)
# print(dset.shape)
# print(dset.flatten())
# print(dset2.flatten()) 




# <HDF5 dataset "kernel:0": shape (128, 1), type "<f4">
#d.shape == (128, 1)
#d[0] == array([-0.14390108], dtype=float32)
# etc.