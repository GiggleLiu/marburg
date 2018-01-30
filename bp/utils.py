import numpy as np
import matplotlib.pyplot as plt

import struct

def unpack(filename):
    with open(filename,'rb') as f:
        _,_, dims = struct.unpack('>HBB',f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        data = np.fromstring(f.read(), dtype=np.uint8).reshape(shape)
        return data

def load_MNIST():
    files = ['MNIST/t10k-images-idx3-ubyte','MNIST/t10k-labels-idx1-ubyte','MNIST/train-images-idx3-ubyte','MNIST/train-labels-idx1-ubyte']
    data = []
    for name in files:
        data.append(unpack(name))
    labels = np.zeros([data[1].shape[0],10])
    for i,iterm in enumerate(data[1]):
        labels[i][iterm] = 1
    data[1] = labels
    labels = np.zeros([data[3].shape[0],10])
    for i,iterm in enumerate(data[3]):
        labels[i][iterm] = 1
    data[3] = labels
    return data

class Buffer(object):
    def __init__(self,data,label):
        self.data=data
        self.label = label
        assert data.shape[0] == label.shape[0]

    def draw(self,batchSize):
        perm = np.random.permutation(self.data.shape[0])
        data = self.data[perm[:batchSize]]
        label = self.label[perm[:batchSize]]
        return data, label