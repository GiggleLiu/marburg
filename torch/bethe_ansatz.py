import os, pdb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from rbm_torch import RBM

class ModifiedBetheAnsatz(nn.Module):
    '''
    Args:
        num_visible (int): number of visible nodes.
        num_hidden (int): number of hidden nodes.

    Attributes:
        W (2darray): weights.
        v_bias (1darray): bias for visible layer.
        h_bias (1darray): bias for hidden layer.
    '''

    def __init__(self, num_visible, num_term, num_hidden):
        super(ModifiedBetheAnsatz, self).__init__()
        self.num_visible = num_visible
        self.W = nn.Parameter(torch.randn(num_term, num_visible) * 1e-1)
        #self.V = nn.Parameter(torch.randn(num_term, num_visible) * 1e-1)
        #self.F1 = nn.Parameter(torch.randn(num_hidden, num_visible) * 1e-1)
        #self.F2 = nn.Parameter(torch.randn(num_term, num_hidden) * 1e-1)
        self.b1 = nn.Parameter(torch.randn(num_term) * 1e-1)
        #self.b2 = nn.Parameter(torch.randn(num_term) * 1e-1)
        self.rbm = RBM(num_visible, num_hidden)

    def f(self, v):
        return F.relu(self.F2.mv(F.relu(self.F1.mv(v)+self.b1))+self.b2)

    def prob_visible(self, v):
        '''
        probability for visible nodes.

        Args:
            v (1darray): visible input.

        Return:
            float: the probability of v.
        '''
        v_ = Variable(v.float())
        if self.W.is_cuda: v_ = v_.cuda()
        return (self.W.mv(v_)+self.b1).sin().sum()*self.rbm.prob_visible(v)

if __name__ == '__main__':
    wf = ModifiedBetheAnsatz(8, 20, 20)
    config = torch.Tensor([-1,1]*4)
    prob = wf.prob_visible(config)
    print(prob)
