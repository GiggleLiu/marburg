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
    def __init__(self):
        pass
    def forward(self,x):
        self.x = x
        return 1.0/(1.0+np.exp(-x))
    def backward(self,delta):
        return delta.dot((1-self.forward(self.x))*self.forward(self.x))
    def update(self):
        pass

class MSE(object):
    def __init__(self):
        pass
    def forward(self,x,l):
        self.x = x
        self.l = l
        return 0.5*((x-l)**2).sum()
    def backward(self,delta):
        return delta*(self.x-self.l)
    def update(self):
        pass

class MLP_layer(object):
    def __init__(self,input_shape,output_shape,activation):
        self.weight = np.random.randn(input_shape,output_shape)
        self.bias = np.random.randn(output_shape)
        self.activation = activation
    def forward(self,x):
        self.x = x
        return self.activation(np.matmul(x,self.weight)+self.bias)
    def backward(self,delta):
        delta = self.activation.backward(delta)
        self.weight_delta = self.x.T.dot(delta)
        self.bias_delta = np.sum(delta,0)
        return self.delta
    def update(self):
        self.weight -= self.weight_delta
        self.bias -= self.bias_delta

class Network(object):
    def __init__(self,layers_list):
        self.layers_list = self.layers_list
    def forward(self,x):
        for layer in self.layers_list:
            x = layer.forward(x)
        return x
    def backward(self,x,learning_rate):
        delta = learning_rate
        for layer in reversed(self.layers_list):
            delta = layer.backward(delta)
    def update(self):
        for layer in self.layers_list:
            layer.update()

def main():
    data = load_MNIST()

if __name__ == "__main__":
    main()