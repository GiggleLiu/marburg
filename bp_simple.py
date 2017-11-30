import numpy as np
import matplotlib.pyplot as plt
import struct

np.random.seed(42)

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

def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

def sigmoid_backward(delta,x):
    return delta*((1-sigmoid(x))*sigmoid(x))

def MSE(x,l):
    return 0.5*((x-l)**2).sum()
def MSE_backward(delta,x):
    return delta*(x)

def main():
    Batch_Size = 5
    Steps = 100
    IMG_Size = 28*28
    Hidden_Size_1 = 110
    Hidden_Size_2 = 10
    Learning_Rate = 0.001
    Maximum = 10000
    Nepochs = 5

    data = load_MNIST()


    weight_1 = np.random.randn(IMG_Size,Hidden_Size_1)
    bias_1 = np.random.randn(Hidden_Size_1)
    
    weight_2 = np.random.randn(Hidden_Size_1,Hidden_Size_2)
    bias_2 = np.random.randn(Hidden_Size_2)
    for _ in range(Nepochs):
        for i in range(100):
            #import pdb
            #pdb.set_trace()
            train = data[0].reshape([-1,IMG_Size])[Batch_Size*(i):Batch_Size*(i+1),:]
            labels = data[1][Batch_Size*(i):Batch_Size*(i+1),:]
            '''
            if i%20 == 0:
                print(labels[0,:])
                print(train[0].reshape([28,28]).shape)
                plt.imshow(train[0].reshape([28,28]))
                plt.show()
            '''
            layer_1_ = train.dot(weight_1)+bias_1
            layer_1 = sigmoid(layer_1_)
            layer_2_ = layer_1.dot(weight_2)+bias_2
            layer_2 = sigmoid(layer_2_)

            loss = MSE(layer_2,labels)
            print(layer_2)
            print(labels)
            #print((layer_2-labels)**2)
            print(loss)

            delta = Learning_Rate
            delta = MSE_backward(delta,layer_2-labels)
            delta = sigmoid_backward(delta,layer_2)
            delta_weight_2 = layer_1.T.dot(delta)
            delta_bias_2 = np.sum(delta,0)
            delta = delta.dot(weight_2.T)

            delta = sigmoid_backward(delta,layer_1)
            delta_weight_1 = train.T.dot(delta)
            delta_bias_1 = np.sum(delta,0)

            weight_1 -= delta_weight_1
            weight_2 -= delta_weight_2
if __name__ == "__main__":
    main()