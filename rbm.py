import os
import pdb
import numpy as np
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt


def show_adn_save(file_name, img):
    npimg = np.transpose(img.numpy(), (1, 2, 0))
    f = "./%s.png" % file_name
    plt.imshow(npimg)
    plt.imsave(f, npimg)


class RBM(nn.Module):
    '''
    Restricted Boltzmann Machine

    Args:
        n_vis (int): number of visible nodes.
        n_hid (int): number of hidden nodes.
    '''

    def __init__(self,
                 n_vis=784,
                 n_hin=500,
                 k=5):
        super(RBM, self).__init__()
        self.W = nn.Parameter(torch.randn(n_hin, n_vis) * 1e-2)
        self.v_bias = nn.Parameter(torch.zeros(n_vis))
        self.h_bias = nn.Parameter(torch.zeros(n_hin))

    def sample_from_p(self, p):
        rand = Variable(torch.rand(p.size()))
        return F.relu(torch.sign(p - rand))

    def v_to_h(self, v):
        p_h = F.sigmoid(F.linear(v, self.W, self.h_bias))
        sample_h = self.sample_from_p(p_h)
        return p_h, sample_h

    def h_to_v(self, h):
        p_v = F.sigmoid(F.linear(h, self.W.t(), self.v_bias))
        sample_v = self.sample_from_p(p_v)
        return p_v, sample_v

    def forward(self, v, k=1):
        '''
        Args:
            k (in): CD-k training, means k times v->h & h->v sweep in a single forward run.
        '''
        pre_h1, h1 = self.v_to_h(v)

        h_ = h1
        for _ in range(k):
            pre_v_, v_ = self.h_to_v(h_)
            pre_h_, h_ = self.v_to_h(v_)

        return v, v_

    def free_energy(self, v):
        vbias_term = v.mv(self.v_bias)
        wx_b = F.linear(v, self.W, self.h_bias)
        hidden_term = wx_b.exp().add(1).log().sum(1)
        return (-hidden_term - vbias_term).mean()

class BinaryRBM(RBM):
    '''
    the binary (-1, 1) version Restricted Boltzmann Machine.
    '''

    def pvh(self, v, h):
        '''
        get the probability of configuration in binary visual/hidden nodes.

        Args:
            v (1darray): visual node configuration.
            h (1darray): hidden node configuration.
        '''
        return F.exp(self.h.mv(self.W.mm(v)) + self.h_bias.dot(h) + self.v_bias.mm(v))

    def pv(self, v):
        '''probability for visual nodes.'''
        return (2 * (self.W.mv(v) + self.h_bias).cosh()).prod() * (self.v_bias.dot(v)).exp()

    def ph(self, h):
        return (2 * (self.W.t().mv(h) + self.v_bias).cosh()).prod() * (self.h_bias.dot(h)).exp()

def train():
    batch_size = 64
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=True, download=not os.path.isdir('data'),
                       transform=transforms.Compose([
                           transforms.ToTensor()
                       ])),
        batch_size=batch_size)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=False, transform=transforms.Compose([
            transforms.ToTensor()
        ])),
        batch_size=batch_size)

    rbm = RBM(k=1)
    train_op = optim.SGD(rbm.parameters(), 0.1)

    for epoch in range(10):
        loss_ = []
        for _, (data, target) in enumerate(train_loader):
            data = Variable(data.view(-1, 784))
            sample_data = data.bernoulli()

            v, v1 = rbm.forward(sample_data, k=1)
            loss = rbm.free_energy(v) - rbm.free_energy(v1)
            loss_.append(loss.data[0])
            print(loss_[-1])
            train_op.zero_grad()
            loss.backward()
            train_op.step()

        print(np.mean(loss_))

    show_adn_save("real", make_grid(v.view(32, 1, 28, 28).data))
    show_adn_save("generate", make_grid(v1.view(32, 1, 28, 28).data))


def test_binary():
    rbm = BinaryRBM(4, 4, k=1)
    pv = rbm.pv(Variable(torch.Tensor([-1, 1, 1, -1])))
    ph = rbm.ph(Variable(torch.Tensor([-1, 1, 1, -1])))
    print(pv, ph)
    pdb.set_trace()


if __name__ == '__main__':
    train()
    #test_binary()
