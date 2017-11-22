'''
Variational Monte Carlo Kernel.
'''
from __future__ import division
import numpy as np
import tensorflow as tf
import pdb

from utils import binning_statistics
from rbm_tf_cd import load_demo_rbm

def vmc_measure(model, initial_config, num_bath=200, num_sample=1000, num_bin=50, measure_step=5):
    '''
    Measure an operator.

    Args:
        model (Model): model definition.
        num_sample (int): number of samples.

    Return:
        number,
    '''
    n_accepted = 0
    print_step = num_sample*measure_step/5

    E_locs, grad_locs = [], []
    config = initial_config
    wf = model.get_wf(config)

    for i in range(num_bath + num_sample*measure_step):
        # generate new config and calculate probability ratio
        config_proposed = model.propose_config(config)
        wf_proposed = model.get_wf(config_proposed)
        prob_ratio = np.abs(wf_proposed/wf).item()**2

        # accept/reject move by one-line metropolis algorithm
        if np.random.random() < prob_ratio:
            config = config_proposed
            wf = wf_proposed
            n_accepted += 1

        # measurements
        if i >= num_bath and i%measure_step==0:
            # here, I choose a lazy way that re-compute on this config, in order to get its gradients easily.
            E_loc, grad_loc = model.local_measure(config)
            E_locs.append(E_loc)
            grad_locs.append(grad_loc)

        # print status
        if i%print_step == print_step-1:
            print('%-10s Accept rate: %.3f' % (i + 1, n_accepted*1. / print_step))
            n_accepted = 0

    # process samples
    E_locs = np.array(E_locs)
    E_mean, E_err = binning_statistics(E_locs, num_bin=num_bin)

    grad_mean = []
    Egrad_mean = []
    for grad_locs_i in zip(*grad_locs):
        grad_mean.append(np.mean(grad_locs_i, axis=0))
        Egrad_mean.append(np.mean(E_locs*grad_locs_i))
    return E_mean.item(), grad_mean, Egrad_mean, E_err.item()

class Model(object):
    def __init__(self, J, rbm, sess, learning_rate=0.01):
        self.J = J
        self.sess = sess
        self.nsite = rbm.visible_input.get_shape()[1].value
        self.rbm = rbm
        self.learning_rate = learning_rate
        self.optimizer = tf.train.AdamOptimizer(learning_rate)

        # build graphs
        self.wf_graph = rbm.build_graph('v-prob')
        self.var_list=[rbm.weights,rbm.hidden_bias,rbm.visible_bias]
        self.gradients = tf.gradients(xs=self.var_list, ys=tf.log(self.wf_graph))
        sess.run(tf.global_variables_initializer())

    def get_wf(self, config):
        wf = self.sess.run(self.wf_graph, feed_dict={self.rbm.visible_input:config[None,:]})
        return wf

    def local_measure(self, config):
        '''get E_loc, grad_loc.'''
        # {d/dW}_{loc}
        res = self.sess.run([self.wf_graph]+self.gradients, feed_dict={self.rbm.visible_input:config[None,:]})
        wf = res[0]
        grad_locs = res[1:]

        # E_{loc}
        E_loc = heisenberg_loc(self.J, config, self.get_wf, wf)

        return E_loc, grad_locs

    def update_vars(self, E_mean, grad_mean, Egrad_mean):
        g_list = [eg - E_mean*g for eg,g in zip(Egrad_mean, grad_mean)]  # no conjugate here
        # assign new values
        assign_ops = [var.assign(var-self.learning_rate*g) for var,g in zip(self.var_list, g_list)]
        #assign_ops = self.optimizer.apply_gradients(zip(g_list, self.var_list))
        self.sess.run(assign_ops)

    def propose_config(self, old_config):
        '''
        flip two positions as suggested spin flips.
        '''
        nsite = len(old_config)
        upmask=old_config==1
        flips=np.random.randint(0,nsite//2,2)
        iflip0=np.where(upmask)[0][flips[0]]
        iflip1=np.where(~upmask)[0][flips[1]]

        config = old_config.copy()
        config[iflip0] = -1
        config[iflip1] = 1
        return config

def heisenberg_loc(J, config, wave_func, wf0):
    '''
    1D Periodic Heisenberg chain local energy.
    '''
    # get weights and flips after applying hamiltonian \sum_i w_i|sigma_i> = H|sigma>
    nsite = len(config)
    wl, flips = [], []
    # J*SzSz terms.
    nn_par = np.roll(config, -1) * config
    wl.append(J / 4. * (nn_par).sum(axis=-1))
    flips.append(np.array([], dtype='int64'))

    # J*SxSx and J*SySy terms.
    mask = nn_par != 1
    i = np.where(mask)[0]
    j = (i + 1) % nsite
    wl += [-J / 2.] * len(i)
    flips.extend(zip(i, j))

    # calculate local energy <psi|H|sigma>/<psi|sigma>
    acc = 0
    for wi, flip in zip(wl, flips):
        config_i = config.copy()
        config_i[list(flip)] *= -1
        eng_i=wi * wave_func(config_i) / wf0
        acc += eng_i
    return acc

def train(model, max_iter=2000):
    '''train a model.'''
    E_list, E_err_list = [], []
    for i in range(max_iter):
        E_mean, grad_mean, Egrad_mean, E_err = vmc_measure(model, initial_config=np.array([-1,1]*(model.nsite//2)), num_sample=1000)
        print('Step %d, energy = %s'%(i, E_mean))
        model.update_vars(E_mean, grad_mean, Egrad_mean)
        E_list.append(E_mean)
        E_err_list.append(E_err)
    return E_list, E_err_list

def run_demo():
    rbm = load_demo_rbm('spin-chain')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        wf_graph = rbm
        model = Model(J=1.0, rbm=rbm, sess=sess, learning_rate=0.05)
        E_list, E_err_list = train(model, 200)
    plt.plot(E_list)

if __name__ == '__main__':
    run_demo()
