import os, pdb, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image

from utils import show_images

class RBM(nn.Module):
    '''
    Restricted Boltzmann Machine

    Args:
        num_visible (int): number of visible nodes.
        num_hidden (int): number of hidden nodes.

    Attributes:
        W (2darray): weights.
        v_bias (1darray): bias for visible layer.
        h_bias (1darray): bias for hidden layer.
    '''

    def __init__(self, num_visible, num_hidden):
        super(RBM, self).__init__()
        self.W = nn.Parameter(torch.randn(num_hidden, num_visible) * 1e-2)
        self.v_bias = nn.Parameter(torch.zeros(num_visible))
        self.h_bias = nn.Parameter(torch.zeros(num_hidden))

    def _v_to_h(self, v):
        '''
        forward pass from visible to hidden, v is visible input.
        '''
        p_h = F.sigmoid(F.linear(v, self.W, self.h_bias))
        sample_h = sample_from_prob(p_h)
        return p_h, sample_h

    def _h_to_v(self, h):
        '''
        backward pass from hidden to visible, h is hidden input.
        '''
        p_v = F.sigmoid(F.linear(h, self.W.t(), self.v_bias))
        sample_v = sample_from_prob(p_v)
        return p_v, sample_v

    def forward(self, v, k=1):
        '''
        Args:
            v (ndarray): visible input.
            k (in): CD-k training, means k times v->h & h->v sweep in a single forward run.

        Returns:
            ndarray: visible obtained through CD sampling.
        '''
        pre_h1, h1 = self._v_to_h(v)

        h_ = h1
        for _ in range(k):
            pre_v_, v_ = self._h_to_v(h_)
            pre_h_, h_ = self._v_to_h(v_)

        return v_

    def free_energy(self, v):
        '''
        free energy E(x) = log(\sum_h exp(x, h)) = log(p(x)*Z).
        It can be used to obtain log-likelihood L = <E(x)>_{data} - <E(x)>_{model}.

        Args:
            v (ndarray): visible input.

        Return:
            float: the free energy loss.
        '''
        vbias_term = v.mv(self.v_bias)
        wx_b = F.linear(v, self.W, self.h_bias)
        hidden_term = wx_b.exp().add(1).log().sum(dim=1)
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

    def prob_visible(self, v):
        '''probability for visible nodes.'''
        return (2 * (self.W.mv(v) + self.h_bias).cosh()).prod() * (self.v_bias.dot(v)).exp()

    def prob_hidden(self, h):
        return (2 * (self.W.t().mv(h) + self.v_bias).cosh()).prod() * (self.h_bias.dot(h)).exp()


def train(use_cuda = False):
    batch_size = 64
    num_visible = 784
    num_hidden = 500

    # create two data loaders to load MNIST data.
    train_loader, test_loader = [torch.utils.data.DataLoader(
        datasets.MNIST('data', train=is_train, download=not os.path.isdir('data'),
                       transform=transforms.Compose([
                           transforms.ToTensor()
                       ])), batch_size=batch_size) for is_train in [True, False]]

    rbm = RBM(num_visible, num_hidden)
    if use_cuda: rbm = rbm.cuda()
    train_op = optim.SGD(rbm.parameters(), 0.1)

    for epoch in range(10):
        loss_ = []
        for step, (data, target) in enumerate(train_loader):
            data = Variable(data.view(-1, 784))
            if use_cuda: data = data.cuda()
            sample_data = data.bernoulli()

            v1 = rbm.forward(sample_data, k=1)
            loss = rbm.free_energy(sample_data) - rbm.free_energy(v1)
            loss_.append(loss.data[0])
            print('step %d, loss = %.4f'%(step, loss_[-1]))

            # zero_grad are needed before backward, otherwise gradients are accumulated.
            train_op.zero_grad()
            loss.backward()

            train_op.step()

        print(np.mean(loss_))

    def _save(file_name, img):
        npimg = np.transpose(img.numpy(), (1, 2, 0))
        f = "data/images/%s.png" % file_name
        plt.imsave(f, npimg)

    _save("real", make_grid(sample_data.view(32, 1, 28, 28).data).cpu())
    _save("generate", make_grid(v1.view(32, 1, 28, 28).data).cpu())


def sample_from_prob(prob_list):
    '''
    from probability to 0-1 sample.

    Args:
        prob_list (1darray): probability of being 1.

    Returns:
        1darray: 0-1 array.
    '''
    rand = Variable(torch.rand(prob_list.size()))
    if prob_list.is_cuda:
        rand = rand.cuda()
    return F.relu(torch.sign(prob_list - rand))

if __name__ == '__main__':
    #show_images(['data/images/%s.png'%x for x in ['real', 'generate']])
    t0=time.time()
    train(use_cuda=True)
    t1=time.time()
    print(t1-t0)
