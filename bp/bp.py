import numpy as np
import matplotlib.pyplot as plt

from utils import load_MNIST
from utils import Buffer
import pdb


class Sigmoid(object):
    def __init__(self):
        self.parameters = []
        self.parameters_deltas = []
    def forward(self,x):
        self.x = x
        self.r =  1.0/(1.0+np.exp(-x))
        return self.r
    def backward(self,delta):
        return delta*((1-self.r)*self.r)

class Softmax(object):
    def __init__(self):
        self.parameters = []
        self.parameters_deltas = []
    def forward(self,x):
        self.x = x
        xtmp = x - x.max(axis=1,keepdims=True)
        exps = np.exp(xtmp)
        self.out = exps / exps.sum(axis=1,keepdims=True)
        return self.out
    def backward(self,delta):
        return delta*self.out - self.out*(delta*self.out).sum(axis=1,keepdims=True)


class MSE(object):
    def __init__(self):
        self.parameters = []
        self.parameters_deltas = []
    def forward(self,x,l):
        self.x = x
        self.l = l
        return 0.5*((x-l)**2).sum()
    def backward(self,delta):
        return delta*(self.x-self.l)

class CrossEntropy(object):
    def __init__(self):
        self.parameters = []
        self.parameters_deltas = []
    def forward(self,x,l):
        x = np.maximum(x,1e-15)
        self.l = l
        self.x = x
        logx = np.log(x)
        y = -l*logx
        return y.sum(-1)

    def backward(self,delta):
        return -delta[...,None]*1./self.x*self.l

class Mean(object):
    def __init__(self):
        self.parameters = []
        self.parameters_deltas = []

    def forward(self,x):
        self.x = x
        return x.mean()

    def backward(self,delta):
        return delta*np.ones(self.x.shape)/np.prod(self.x.shape)

class Linear(object):
    def __init__(self,input_shape,output_shape,mean = 0, variance = 0.1):
        self.parameters = [mean+variance*np.random.randn(input_shape,output_shape),mean+variance*np.random.randn(output_shape)]
        self.parameters_deltas = [None,None]
    def forward(self,x):
        self.x = x
        return np.matmul(x,self.parameters[0])+self.parameters[1]
    def backward(self,delta):
        self.parameters_deltas[0] = self.x.T.dot(delta)
        self.parameters_deltas[1] = np.sum(delta,0)
        return delta.dot(self.parameters[0].T)

def main():
    np.random.seed(42)
    import argparse

    parser = argparse.ArgumentParser(description='')
    group = parser.add_argument_group('learning  parameters')
    group.add_argument("-batchsize",type=int,default=100,help="size of batch to train")
    group.add_argument("-lr", type=float, default=0.5, help="learning rate")

    group = parser.add_argument_group('test parameters')
    group.add_argument("-testbatch",type=int,default=100,help="num of test samples")
    args = parser.parse_args()

    data = load_MNIST()
    buff = Buffer(data[2],data[3])
    testbuff = Buffer(data[0],data[1])

    l1 = Linear(784,10)

    activation1 = Softmax()

    losslayer = CrossEntropy()
    mean = Mean()

    layers = [l1,activation1]

    iterations = buff.data.shape[0]//args.batchsize

    for j in range(iterations):
        #img = (train/255.0).reshape(args.batchsize,-1)
        train,label = buff.draw(args.batchsize)
        train = (train/255.0).reshape(args.batchsize,-1)
        img = train
        tmp = l1.forward(img)
        result = activation1.forward(tmp)

        l = losslayer.forward(result,label)
        l = mean.forward(l)

        label_p = np.argmax(result,axis=1)
        label_t = np.argmax(label,axis=1)
        ratio = np.sum(label_p == label_t)/label_t.shape[0]

        print("iteration:",j,"/",iterations,"loss:",l,"ratio:",ratio)

        delta = mean.backward(args.lr)
        delta = losslayer.backward(delta)

        delta = activation1.backward(delta)
        delta = l1.backward(delta)

        l1.parameters[0] = l1.parameters[0]-l1.parameters_deltas[0]
        l1.parameters[1] = l1.parameters[1]-l1.parameters_deltas[1]

    test,label = testbuff.draw(args.testbatch)

    result = (test/255.0).reshape(args.testbatch,-1)

    for layer in layers:
        result = layer.forward(result)

    label_p = np.argmax(result,axis=1)
    label_t = np.argmax(label,axis=1)
    ratio = np.sum(label_p == label_t)/label_t.shape[0]

    l = losslayer.forward(result,label)/args.testbatch

    print(l)
    print(ratio)

    pdb.set_trace()


if __name__ == "__main__":
    main()
