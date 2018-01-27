'''
Variational Monte Carlo Kernel.
'''
from __future__ import division
import numpy as np
import pdb

__all__ = ['vmc_measure', 'binning_statistics']

def vmc_measure(model, initial_config, num_bath=200, num_sample=1000, num_bin=50, measure_step=None):
    '''
    Measure an operator.

    Args:
        model (Model): model definition.
        num_sample (int): number of samples.

    Return:
        number,
    '''
    if measure_step is None:
        measure_step = len(initial_config)
    print_step = num_sample * measure_step // 5

    energy_loc_list, grad_loc_list = [], []
    config = initial_config
    wf = model.psi(config)

    n_accepted = 0
    for i in range(num_bath + num_sample * measure_step):
        # generate new config and calculate probability ratio
        config_proposed = model.propose_config(config)
        wf_proposed = model.psi(config_proposed)
        prob_ratio = np.abs(wf_proposed / wf).item()**2

        # accept/reject move by one-line metropolis algorithm
        if np.random.random() < prob_ratio:
            config = config_proposed
            wf = wf_proposed
            n_accepted += 1

        # measurements
        if i >= num_bath and i % measure_step == 0:
            # here, I choose a lazy way that re-compute on this config, in order to get its gradients easily.
            energy_loc, grad_loc = model.local_measure(config)
            energy_loc_list.append(energy_loc)
            grad_loc_list.append(grad_loc)

        # print status
        if i % print_step == print_step - 1:
            print('%-10s Accept rate: %.3f' %
                  (i + 1, n_accepted * 1. / print_step))
            n_accepted = 0

    # process samples
    energy_loc_list = np.array(energy_loc_list)
    energy, energy_precision = binning_statistics(energy_loc_list, num_bin=num_bin)

    grad_mean = []
    energy_grad = []
    gradgrad_mean = []
    for grad_loc in zip(*grad_loc_list):
        grad_loc = np.array(grad_loc)
        grad_mean.append(grad_loc.mean(axis=0))
        energy_grad.append(
            (energy_loc_list[(slice(None),) + (None,) * (grad_loc.ndim - 1)] * grad_loc).mean(axis=0))
    return energy, grad_mean, energy_grad, energy_precision


def binning_statistics(var_list, num_bin):
    '''
    binning statistics for variable list.
    '''
    num_sample = len(var_list)
    var_list = var_list.real
    if num_sample % num_bin != 0:
        raise
    size_bin = num_sample // num_bin

    # mean, variance
    mean = np.mean(var_list, axis=0)
    variance = np.var(var_list, axis=0)

    # binned variance and autocorrelation time.
    variance_binned = np.var(
        [np.mean(var_list[size_bin * i:size_bin * (i + 1)]) for i in range(num_bin)])
    t_auto = 0.5 * size_bin * \
        np.abs(np.mean(variance_binned) / np.mean(variance))
    stderr = np.sqrt(variance_binned / num_bin)
    print('Binning Statistics: Energy = %.4f +- %.4f, Auto correlation Time = %.4f' %
          (mean, stderr, t_auto))
    return mean, stderr

