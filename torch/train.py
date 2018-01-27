'''
Variational Monte Carlo Kernel.
'''
from __future__ import division
import numpy as np
import torch
import matplotlib.pyplot as plt
import pdb

from hamiltonians import heisenberg_loc
from rbm_torch import RBM
from vmc import vmc_measure

def train(model, learning_rate):
    '''
    train a model.

    Args:
        model (obj): a model that meet VMC model definition.
    '''
    initial_config = np.array([0, 1] * (model.num_spin // 2))

    while True:
        # get expectation values for energy, gradient and their product,
        # as well as the precision of energy.
        energy, grad, energy_grad, precision = vmc_measure(
            model, initial_config=initial_config, num_sample=500)

        # update variables using steepest gradient descent
        g_list = [eg - energy * g for eg, g in zip(energy_grad, grad)]
        for var, g in zip(model.ansatz.parameters(), g_list):
            var.data -= learning_rate * torch.from_numpy(g).float()
        yield energy, precision


class VMCKernel(object):
    '''
    variational monte carlo kernel.
    '''
    def __init__(self, energy_loc, ansatz):
        self.ansatz = ansatz
        self.energy_loc = energy_loc

    @property
    def num_spin(self):
        return self.ansatz.v_bias.size(0)

    def psi(self, config):
        '''
        query the wavefunction.

        Args:
            config (1darray): the bit string as a configuration.

        Returns:
            number: the projection of wave function on config, i.e. <config|psi>.
        '''
        psi = self.ansatz.prob_visible(torch.from_numpy(config))
        return psi

    def prob(self, config):
        '''probability of configuration.'''
        return abs(self.psi(config).data[0])**2

    def local_measure(self, config):
        '''
        get local quantities energy_loc, grad_loc.

        Args:
            config (1darray): the bit string as a configuration.

        Returns:
            number, list: local energy and local gradients for variables.
        '''
        psi_loc = self.psi(config)

        # get gradients {d/dW}_{loc}
        self.ansatz.zero_grad()
        psi_loc.backward()
        grad_loc = [p.grad.data.numpy()/psi_loc.data.numpy() for p in self.ansatz.parameters()]

        # E_{loc}
        eloc = self.energy_loc(config, lambda x: self.psi(x).data, psi_loc.data)[0]
        return eloc, grad_loc

    @staticmethod
    def propose_config(old_config, prob_flip=0.05):
        '''
        flip two positions as suggested spin flips.
        '''
        # take ~ 5% probability to flip all spin, can making VMC sample better in Heisenberg model
        if np.random.random() < prob_flip:
            return 1-old_config

        num_spin = len(old_config)
        upmask = old_config == 0
        flips = np.random.randint(0, num_spin // 2, 2)
        iflip0 = np.where(upmask)[0][flips[0]]
        iflip1 = np.where(~upmask)[0][flips[1]]

        config = old_config.copy()
        config[iflip0] = 1
        config[iflip1] = 0
        return config


def run_demo():
    seed = 10086
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    max_iter = 200
    num_spin = 8
    num_hidden = 20

    # visualize the loss history
    energy_list, precision_list = [], []
    def _update_curve(energy, precision):
        energy_list.append(energy)
        precision_list.append(precision)
        plt.errorbar(np.arange(1, len(energy_list) + 1), energy_list, yerr=precision_list)

    rbm = RBM(num_spin, num_hidden)
    model = VMCKernel(heisenberg_loc, ansatz=rbm)

    E_exact=-3.65109341
    for i, (energy, precision) in enumerate(train(model, learning_rate = 0.1)):
        _update_curve(energy, precision)
        print('Step %d, dE = %.4f' % (i, energy - E_exact))

        # stop condition
        if i >= max_iter:
            break

if __name__ == '__main__':
    run_demo()
