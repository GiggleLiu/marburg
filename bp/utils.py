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

def normalization(data):
    return data+np.random.uniform(size=data.shape)/256.0

class Buffer(object):
    def __init__(self,maximum,data=None,testRatio=0.0):
        testSize = int(testRatio*maximum)
        self.testRatio = testRatio
        self.capacity = int(maximum*(1+self.testRatio))
        self.data=data
        if data is None:
            self.maximum = 0
        else:
            self.maximum = len(data)

    def draw(self,batchSize,testSize=None):
        maximum = int(self.maximum*(1.0-self.testRatio))
        if batchSize > maximum:
            batchSize = maximum
        perm = np.random.permutation(maximum)
        if testSize is None:
            return self.data[perm[:batchSize]]
        else:
            if testSize > self.maximum-maximum:
                testSize = self.maximum-maximum
            permp = np.random.permutation(self.maximum-maximum)+maximum
            train = self.data[perm[:batchSize]]
            test = self.data[permp[-testBatchSize:-1]]
            return train, test
    def drawtest(self,testSize):
        if testSize > int(self.maximum*(self.testRatio)):
            testSize = int(self.maximum*(self.testRatio))
        perm = np.random.permutation(self.maximum-maximum)+maximum
        test = self.data[perm]
        return test
    def push(self,data):
        if self.data is None:
            self.data = data
        else:
            self.data = torch.cat((self.data,data),0)
        if self.data.shape[0] > self.capacity:
            self._maintain()
        self.maximum = len(self.data)
    def _maintain(self):
        perm = np.random.permutation(self.data.shape[0])
        self.data = self.data[perm[:self.capacity]]