import numpy as np
import matplotlib.pyplot as plt

import struct
import subprocess
import os

def unpack(filename):
    with open(filename,'rb') as f:
        _,_, dims = struct.unpack('>HBB',f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        data = np.fromstring(f.read(), dtype=np.uint8).reshape(shape)
        return data

def load_MNIST():
    path = "../data/raw/"
    files = ['t10k-images-idx3-ubyte','t10k-labels-idx1-ubyte','train-images-idx3-ubyte','train-labels-idx1-ubyte']
    data = []
    for name in files:
        name = path+name
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

def numdiff(layer, x, var, dy, delta):
    '''numerical differenciation.'''
    var_raveled = var.ravel()

    var_delta_list = []
    for ix in range(len(var_raveled)):
        var_raveled[ix] += delta/2.
        yplus = layer.forward(x)
        var_raveled[ix] -= delta
        yminus = layer.forward(x)
        var_delta = ((yplus - yminus)/delta*dy).sum()
        var_delta_list.append(var_delta)

        # restore changes
        var_raveled[ix] += delta/2.
    return np.array(var_delta_list)

def sanity_check(layer, x, args, delta=0.01, precision=1e-3):
    '''
    perform sanity check for a layer,
    raise an assertion error if failed to pass all sanity checks.

    Args:
        layer (obj): user defined neural network layer.
        x (ndarray): input array.
        args: additional arguments for forward function.
        delta: the strength of perturbation used in numdiff.
        precision: the required precision of gradient (usually introduced by numdiff).
    '''
    y = layer.forward(x, *args)
    dy = np.random.randn(y.shape)
    x_delta = layer.backward(dy)

    for var, var_delta in zip([x] + layer.parameters, [dx] + layer.parameters_deltas):
        x_delta_num = numdiff(layer, x, var, dy, delta)
        assert(np.all(abs(x_delta_num - var_delta) < precision))

def download_MNIST():
    base = "http://yann.lecun.com/exdb/mnist/"
    objects = ['t10k-images-idx3-ubyte','t10k-labels-idx1-ubyte','train-images-idx3-ubyte','train-labels-idx1-ubyte']
    end = ".gz"
    path = "../data/raw/"
    cmd = ["mkdir","-p",path]
    subprocess.check_call(cmd)
    for obj in objects:
        if not os.path.isfile(path+obj):
            cmd = ["wget",base+obj+end,"-P",path]
            subprocess.check_call(cmd)
            cmd = ["gzip","-d",path+obj+end]
            subprocess.check_call(cmd)

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