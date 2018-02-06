'''
Variational Monte Carlo Kernel.
'''
from __future__ import division
import numpy as np
import torch, time
import matplotlib.pyplot as plt
import pdb

from hamiltonians import heisenberg_loc, J1J2_loc
from rbm_torch import RBM
from vmc import vmc_measure
from profilehooks import profile

def train(model, learning_rate, use_cuda):
    '''
    train a model.

    Args:
        model (obj): a model that meet VMC model definition.
        learning_rate (float): the learning rate for SGD.
    '''
    #initial_config = torch.Tensor([-1, 1] * (model.ansatz.num_visible // 2))
    initial_config = np.array([-1, 1] * (model.ansatz.num_visible // 2))
    if use_cuda:
        #initial_config = initial_config.cuda()
        model.ansatz.cuda()

    while True:
        # get expectation values for energy, gradient and their product,
        # as well as the precision of energy.
        energy, grad, energy_grad, precision = vmc_measure(
            model, initial_config=initial_config, num_sample=500)

        # update variables using steepest gradient descent
        g_list = [eg - energy * g for eg, g in zip(energy_grad, grad)]
        for var, g in zip(model.ansatz.parameters(), g_list):
            delta = learning_rate * g
            var.data -= delta
        yield energy, precision


class VMCKernel(object):
    '''
    variational monte carlo kernel.

    Attributes:
        energy_loc (func): local energy <\sigma|H|\psi>/<\sigma|\psi>.
        ansatz (Module): torch neural network.
    '''
    def __init__(self, energy_loc, ansatz):
        self.ansatz = ansatz
        self.energy_loc = energy_loc

    def psi(self, config):
        '''
        query the wavefunction.

        Args:
            config (1darray): the bit string as a configuration.

        Returns:
            Variable: the projection of wave function on config, i.e. <config|psi>.
        '''
        psi = self.ansatz.prob_visible(torch.from_numpy(config))
        #psi = self.ansatz.prob_visible(config)
        return psi

    def prob(self, config):
        '''
        probability of configuration.

        Args:
            config (1darray): the bit string as a configuration.

        Returns:
            number: probability |<config|psi>|^2.
        '''
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
        grad_loc = [p.grad.data/psi_loc.data[0] for p in self.ansatz.parameters()]

        # E_{loc}
        eloc = self.energy_loc(config, lambda x: self.psi(x).data, psi_loc.data)[0]
        return eloc, grad_loc

    @staticmethod
    def propose_config(old_config, prob_flip=0.05):
        '''
        flip two positions as suggested spin flips.

        Args:
            old_config (1darray): spin configuration, which is a [-1,1] string.
            prob_flip (float): the probability to flip all spins, to make VMC more statble in Heisenberg model.

        Returns:
            1darray: new spin configuration.
        '''
        # take ~ 5% probability to flip all spin, can making VMC sample better in Heisenberg model
        if np.random.random() < prob_flip:
            return -old_config

        #is_cuda = old_config.is_cuda
        #old_config = old_config.cpu().numpy()
        num_spin = len(old_config)
        upmask = old_config == 1
        flips = np.random.randint(0, num_spin // 2, 2)
        iflip0 = np.where(upmask)[0][flips[0]]
        iflip1 = np.where(~upmask)[0][flips[1]]

        config = old_config.copy()
        config[iflip0] = -1
        config[iflip1] = 1
        #config = torch.from_numpy(config)
        #if is_cuda: config = config.cuda()
        return config

def run_demo():
    seed = 10086
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    max_iter = 200
    num_spin = 8
    num_hidden = 50

    # visualize the loss history
    energy_list, precision_list = [], []
    def _update_curve(energy, precision):
        energy_list.append(energy)
        precision_list.append(precision)
        plt.errorbar(np.arange(1, len(energy_list) + 1), energy_list, yerr=precision_list)

    rbm = RBM(num_spin, num_hidden)
    model = VMCKernel(heisenberg_loc, ansatz=rbm)

    E_exact = -3.65109341
    t0 = time.time()
    for i, (energy, precision) in enumerate(train(model, learning_rate = 0.1, use_cuda = True)):
        t1 = time.time()
        print('Step %d, dE/|E| = %.4f, elapse = %.4f' % (i, -(energy - E_exact)/E_exact, t1-t0))
        _update_curve(energy, precision)
        t0 = time.time()

        # stop condition
        if i >= max_iter:
            break

def run_solution():
    seed = 10086
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    max_iter = 200
    num_spin = 20
    num_hidden = 4
    J2 = 0.0

    # visualize the loss history
    energy_list, precision_list = [], []
    def _update_curve(energy, precision):
        energy_list.append(energy)
        precision_list.append(precision)
        plt.errorbar(np.arange(1, len(energy_list) + 1), energy_list, yerr=precision_list)

    from bethe_ansatz import ModifiedBetheAnsatz
    bethe = ModifiedBetheAnsatz(num_spin, 1, num_hidden)
    model = VMCKernel(lambda a,b,c: J1J2_loc(a,b,c,1,1,J2,J2,True), ansatz=bethe)

    E_exact = -8.90439
    t0 = time.time()
    for i, (energy, precision) in enumerate(train(model, learning_rate = 0.1, use_cuda = False)):
        t1 = time.time()
        print('Step %d, dE/|E| = %.4f, elapse = %.4f' % (i, -(energy - E_exact)/E_exact, t1-t0))
        _update_curve(energy, precision)
        t0 = time.time()

        # stop condition
        if i >= max_iter:
            break

if __name__ == '__main__':
    run_solution()
