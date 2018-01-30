import numpy as np
import matplotlib.pyplot as plt

from utils import load_MNIST
from utils import Buffer


class Sigmoid(object):
    def forward(self,x):
        self.x = x
        self.r =  1.0/(1.0+np.exp(-x))
        return self.r
    def backward(self,delta):
        return delta*((1-self.r)*self.r)

class MSE(object):
    def forward(self,x,l):
        self.x = x
        self.l = l
        return 0.5*((x-l)**2).sum()
    def backward(self,delta):
        return delta*(self.x-self.l)

class Linear(object):
    def __init__(self,input_shape,output_shape):
        self.weight = np.random.randn(input_shape,output_shape)
        self.bias = np.random.randn(output_shape)
    def forward(self,x):
        self.x = x
        return np.matmul(x,self.weight)+self.bias
    def backward(self,delta):
        self.weight_delta = self.x.T.dot(delta)
        self.bias_delta = np.sum(delta,0)
        return delta.dot(self.weight.T)

class Network(object):
    def __init__(self,layers_list,loss):
        self.layers_list = layers_list
        self.lossfn = loss
    def forward(self,x):
        for layer in self.layers_list:
            x = layer.forward(x)
        return x
    def loss(self,x,label):
        x = self.forward(x)
        l = self.lossfn.forward(x,label)
        return l
    def backward(self,learning_rate):
        delta = self.lossfn.backward(learning_rate)
        for layer in reversed(self.layers_list):
            delta = layer.backward(delta)
    def update(self):
        for layer in self.layers_list:
            if type(layer) == Linear:
                layer.weight -= layer.weight_delta
                layer.bias -= layer.bias_delta

def normalization(data):
    return (data+np.random.uniform(size=data.shape))/256.0

def main():
    import argparse
    parser = argparse.ArgumentParser(description='')
    group = parser.add_argument_group('learning  parameters')
    group.add_argument("-epochs",type=int,default=10,help="epochs to run")
    group.add_argument("-batchsize",type=int,default=300,help="size of batch to train")
    group.add_argument("-lr", type=float, default=0.001, help="learning rate")
    group.add_argument("-iterations", type = int,default=-1,help="num of itrations in each epoch")

    group = parser.add_argument_group('network structure')
    group.add_argument("-layers",type=int,default=5,help="layers in network")
    group.add_argument("-hidden",type=int,default=1000,help="hidden features in each layer")

    group = parser.add_argument_group('test parameters')
    group.add_argument("-testbatch",type=int,default=100,help="num of test samples")
    args = parser.parse_args()

    data = load_MNIST()
    buff = Buffer(data[2],data[3])
    testbuff = Buffer(data[0],data[1])

    l1 = Linear(784,500)
    l2 = Linear(500,100)
    l3 = Linear(100,10)

    activation1 = Sigmoid()
    activation2 = Sigmoid()
    activation3 = Sigmoid()

    mse = MSE()

    network = Network([l1,activation1,l2,activation2,l3,activation3],mse)

    if args.iterations == -1:
        iterations = buff.data.shape[0]//args.batchsize
    else:
        iterations = args.iterations

    for i in range(args.epochs):
        for j in range(iterations):
            train,label = buff.draw(args.batchsize)

            train = normalization(train)

            l = network.loss(train.reshape(train.shape[0],-1),label)

            print("epoch:",i,"iteration:",j,"/",iterations,"loss:",l)

            network.backward(args.lr)
            network.update()

    test,label = testbuff.draw(args.testbatch)

    test = normalization(test)

    r = network.forward(test)

    l = mse.forward(r,label)

    print(l)

if __name__ == "__main__":
    main()
