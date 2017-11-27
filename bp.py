import numpy as np
import matplotlib.pyplot as plt

import struct

def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

def softmax(x):
    exp = np.exp(x)
    return exp/exp.sum()

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
    print(data[1].shape[0])
    labels = np.zeros([data[1].shape[0],10])
    for i,iterm in enumerate(data[1]):
        labels[i][iterm] = 1
    data[1] = labels
    labels = np.zeros([data[3].shape[0],10])
    for i,iterm in enumerate(data[3]):
        labels[i][iterm] = 1
    data[3] = labels
    return data

class MLP_layer(object):
    def __init__(self,input_shape,output_shape,activation):
        self.weight = np.random.randn(input_shape,output_shape)
        self.bias = np.random.randn(output_shape)
        self.activation = activation
    def forward(self,x):
        return self.activation(np.matmul(x,self.weight)+self.bias)
    def backward(self,x):
        pass

def main():
    data = load_MNIST()

if __name__ == "__main__":
    main()