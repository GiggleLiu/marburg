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

class Sigmoid(object):
    @staticmethod
    def forward(x):
        return 1.0/(1.0+np.exp(-x))
    @staticmethod
    def backward(x,delta):
        return delta.dot((1-self.forward(x))*self.forward(x))
    @staticmethod
    def update():
        pass

class MSE(object):
    @staticmethod
    def forward(x,l):
        return 0.5*((x-l)**2).sum()
    @staticmethod
    def backward(x,l,delta):
        return delta*(x-l)
    @staticmethod
    def update():
        pass

class MLP_layer(object):
    def __init__(self,input_shape,output_shape,activation):
        self.weight = np.random.randn(input_shape,output_shape)
        self.bias = np.random.randn(output_shape)
        self.activation = activation
    def forward(self,x):
        self.before_activation = np.matmul(x,self.weight)+self.bias)
        return self.activation(self.before_activation)
    def backward(self,x,delta):
        delta = self.activation.backward(self.before_activation,delta)
        self.weight_delta = x.T.dot(delta)
        self.bias_delta = np.sum(delta,0)
        return self.delta
    def update():
        self.weight -= self.weight_delta
        self.bias -= self.bias_delta

class Network(object):
    def __init__(self,layers_list):
        self.layers_list = self.layers_list
        self.memory=[]
    def forward(self,x):
        for layer in self.layers_list:
            self.memory.append(x)
            x = layer.forward(x)
        return x
    def backward(self,x,learning_rate):
        delta = learning_rate
        for layer in reversed(self.layers_list):
            delta = layer.backward(delta)
    def SGD():
        pass

def main():
    data = load_MNIST()

if __name__ == "__main__":
    main()