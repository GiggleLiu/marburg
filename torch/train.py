'''
Variational Monte Carlo Kernel.
'''
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import pdb

from utils import binning_statistics


def train(model, max_iter=2000, learning_rate=0.1):
    '''train a model.'''
    initial_config = np.array([0, 1] * (model.nsite // 2))

    for i in range(max_iter):
        # get expectation values for energy, gradient and their product,
        # as well as the precision of energy.
        energy, grad, energy_grad, precision = vmc_measure(
            model, initial_config=initial_config, num_sample=500)
        print('Step %d, dE = %.4f' % (i, energy - model.E_exact))

        # update variables using steepest gradient descent
        g_list = [eg - energy * g for eg, g in zip(energy_grad, grad)]
        for var, g in zip(self.var_list, g_list):
            var -= learning_rate * g
        yield energy, precision


class VMCKernel(object):
    '''
    variational monte carlo kernel.
    '''
    def __init__(self, energy_loc_func, rbm, E_exact, use_cnn=True):
        self.nsite = rbm.visible_input.get_shape()[1].value
        self.rbm = rbm
        self.E_exact = E_exact

        # build graphs
        if use_cnn:
            self.wf_graph = rbm.build_graph('psi-cnn')
            self.var_list = [rbm.weights, rbm.hidden_bias]
        else:
            self.wf_graph = rbm.build_graph('psi')
            self.var_list = [rbm.weights, rbm.hidden_bias, rbm.visible_bias]
        self.gradients = tf.gradients(
            xs=self.var_list, ys=tf.log(self.wf_graph))

    def psi(self, config):
        '''
        query the wavefunction.

        Args:
            config (1darray): the bit string as a configuration.

        Returns:
            number: the projection of wave function on config, i.e. <config|psi>.
        '''
        psi = self.sess.run(self.wf_graph, feed_dict={
                           self.rbm.visible_input: config[None, :]})
        return psi

    def local_measure(self, config):
        '''
        get local quantities energy_loc, grad_loc.

        Args:
            config (1darray): the bit string as a configuration.

        Returns:
            number, list: local energy and local gradients for variables.
        '''
        # {d/dW}_{loc}
        psi_loc = self.psi(config)
        psi_loc.zero_grad()
        psi_loc.backward()
        # take gradients out
        grad_loc = xx

        # E_{loc}
        energy_loc = self.energy_loc_func(config, self.psi, psi_loc)

        return energy_loc, grad_loc

    def propose_config(old_config, prob_flip=0.05):
        '''
        flip two positions as suggested spin flips.
        '''
        # take ~ 5% probability to flip all spin, can making VMC sample better in Heisenberg model
        if np.random.random() < prob_flip:
            return 1-old_config

        nsite = len(old_config)
        upmask = old_config == 0
        flips = np.random.randint(0, nsite // 2, 2)
        iflip0 = np.where(upmask)[0][flips[0]]
        iflip1 = np.where(~upmask)[0][flips[1]]

        config = old_config.copy()
        config[iflip0] = 1
        config[iflip1] = 0
        return config


def run_demo():
    tf.set_random_seed(10087)
    np.random.seed(10086)

    model = VMCKernel(heisenberg_loc, rbm=rbm, sess=sess, E_exact=-3.65109341)
    energy_list, precision_list = train(model, max_iter=200)
    plt.errorbar(arange(1, len(energy_list) + 1), energy_list, yerr=precision_list)


if __name__ == '__main__':
    run_demo()
