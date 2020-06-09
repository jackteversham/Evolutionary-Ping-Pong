import h5py
import numpy as np


f = h5py.File('weightFiles/weights3.h5', 'r')
f2 =  h5py.File('weightFiles/weights4.h5', 'r')
#f = h5py.File('james_model_weights.h5', 'r')
print(list(f.attrs))

print(list(f.keys())) #h5 file acts like a dictionary => this gives the keys
grp1 = f['dense_4']
print(list(grp1.keys()))

# will get a list of layer names which you can use as index
grps = f['dense_4/dense_4/']
print(list(grps.keys()))


dset = f['dense_4/dense_4/kernel:0']
dset2 = f2['dense_4/dense_4/kernel:0']
dset = np.array(dset)
#print(dset.name)
print(dset.shape)
print(dset[0:10])
print(dset2[0:10])




# <HDF5 dataset "kernel:0": shape (128, 1), type "<f4">
#d.shape == (128, 1)
#d[0] == array([-0.14390108], dtype=float32)
# etc.