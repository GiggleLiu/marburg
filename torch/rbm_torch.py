import os, pdb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

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
        self.W = nn.Parameter(torch.randn(num_hidden, num_visible) * 1e-1)
        self.v_bias = nn.Parameter(torch.zeros(num_visible))
        self.h_bias = nn.Parameter(torch.randn(num_hidden) * 1e-1)

    def _v_to_h(self, v):
        '''
        forward pass p(h|v) from visible to hidden, v is visible input.
        '''
        p_h = F.sigmoid(F.linear(v, self.W, self.h_bias))
        return p_h

    def _h_to_v(self, h):
        '''
        backward pass p(v|h) from hidden to visible, h is hidden input.
        '''
        p_v = F.sigmoid(F.linear(h, self.W.t(), self.v_bias))
        return p_v

    def contrastive_divergence(self, v, k=1):
        '''
        Args:
            v (ndarray): visible input.
            k (in): CD-k training, means k times v->h & h->v sweep in a single contrastive divergence run.

        Returns:
            ndarray: visible obtained through CD sampling.
        '''
        prob_h = self._v_to_h(v)
        h = sample_from_prob(prob_h)
        for _ in range(k):
            prob_v = self._h_to_v(h)
            v = sample_from_prob(prob_v)
            prob_h = self._v_to_h(v)
            h = sample_from_prob(prob_h)

        return v

    def free_energy(self, v):
        '''
        free energy E(x) = log(\sum_h exp(x, h)) = log(p(x)*Z).
        It can be used to obtain log-likelihood L = <E(x)>_{data} - <E(x)>_{model}.

        Args:
            v (1darray,2darray): visible input with size ([batch_size, ]data_size).

        Return:
            float: the free energy loss.
        '''
        vbias_term = v.mv(self.v_bias)
        wx_b = F.linear(v, self.W, self.h_bias)
        hidden_term = wx_b.exp().add(1).log().sum(dim=-1)
        return (-hidden_term - vbias_term).mean()

    def prob_visible(self, v):
        '''
        probability for visible nodes.

        Args:
            v (1darray): visible input.

        Return:
            float: the probability of v.
        '''
        v = Variable(v.float())
        return (2 * (self.W.mv(v) + self.h_bias).cosh()).prod() * (self.v_bias.dot(v)).exp()
        #v = Variable(v.float()[None,:])
        #return self.free_energy(v).exp()

def mnist_train(use_cuda, network_file):
    '''
    train RBM on MNIST dataset.
    '''
    num_visible = 784
    num_hidden = 500

    rbm = RBM(num_visible, num_hidden)
    if use_cuda: rbm = rbm.cuda()
    loader = mnist01_loader(True, use_cuda, batch_size=64)

    train_op = torch.optim.SGD(rbm.parameters(), 0.1)

    for epoch in range(10):
        loss_list = []
        for data, label in loader():
            v1 = rbm.contrastive_divergence(data, k=1)
            loss = rbm.free_energy(data) - rbm.free_energy(v1)
            loss_list.append(loss.data[0])

            # get gradients and update parameters using gradients.
            # zero_grad are needed before backward, otherwise gradients are accumulated.
            train_op.zero_grad()
            loss.backward()
            train_op.step()

        torch.save(rbm.state_dict(), network_file)
        print('epoch %d, Mean Loss = %.4f'%(epoch, np.mean(loss_list)))

def mnist_analyse(network_file):
    '''
    train RBM on MNIST dataset.
    '''
    num_visible = 784
    num_hidden = 500

    rbm = RBM(num_visible, num_hidden)
    rbm.load_state_dict(torch.load(network_file))

    # generate data
    data, label = next(mnist01_loader(False, False, batch_size=32)())
    generated = rbm.contrastive_divergence(data, k=1)

    gs = plt.GridSpec(2,1)
    for irow, img in [(0, data), (1, generated)]:
        img_grid = make_grid(img.view(32, 1, 28, 28).data)
        npimg = np.transpose(img_grid.numpy(), (1, 2, 0))

        plt.subplot(gs[irow,0])
        plt.imshow(npimg)
        plt.axis('off')
    plt.show()

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

def mnist01_loader(is_train, use_cuda, batch_size):
    '''
    yield image and label from mnist dataset.

    Args:
        is_train (bool): yield traning set if True, else test set.
        use_cuda (bool): return data on GPU in True.
        batch_size (int): size of a batch.

    Returns:
        func: an iterator function.
    '''
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=is_train,
                    download=not os.path.isdir('data'),
                       transform=transforms.Compose([
                           transforms.ToTensor()
                       ])), batch_size=batch_size, pin_memory=True)

    def iterator():
        for data, label in test_loader:
            # transform to binary mnist image
            data = Variable(data.view(-1, 784))
            data = data.bernoulli()
            if use_cuda:
                # copy data to gpu memory
                data = data.cuda()
                label = label.cuda()
            yield data, label
    return iterator

if __name__ == '__main__':
    network_file = 'data/rbm-mnist.dat'
    #mnist_train(use_cuda=True, network_file=network_file)
    mnist_analyse('data/rbm-mnist.dat')
