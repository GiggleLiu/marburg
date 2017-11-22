import numpy as np
import tensorflow as tf

__all__ = ['sample_prob', 'typed_normal', 'save_img44']

def sample_prob(probs, name):
    '''give 1 with specific probability, 0 otherwise.'''
    with tf.variable_scope(name):
        return tf.nn.relu(tf.sign(probs - tf.cast(tf.random_uniform(probs.shape, 0,1), probs.dtype)))

def typed_normal(dtype, shape, scale):
    '''normal distribution with specific data type.'''
    dtype = dtype.name
    if dtype in ['complex64', 'complex128']:
        arr = np.random.normal(0,scale,shape) + 1j*np.random.normal(0,scale,shape)
    else:
        arr = np.random.normal(0, scale, shape)
    return arr.astype(dtype)

def save_img44(filename, data):
    import matplotlib.pyplot as plt
    nrow, ncol = 4, 4
    gs = plt.GridSpec(nrow,ncol)
    fig = plt.figure(figsize = (6,5))
    for i in range(nrow):
        for j in range(ncol):
            ax = plt.subplot(gs[i,j])
            ax.imshow(data[i*ncol+j], interpolation='none', cmap='Greys')
            ax.axis('equal')
            ax.axis('off')
    gs.tight_layout(fig, pad=0)
    plt.show()

def binning_statistics(var_list, num_bin):
    '''
    binning statistics for variable list.
    '''
    num_sample = len(var_list)
    if num_sample%num_bin!=0:
        raise
    size_bin = num_sample//num_bin

    # mean, variance
    mean = np.mean(var_list, axis=0)
    variance= np.var(var_list, axis=0)

    # binned variance and autocorrelation time.
    variance_binned = np.var([np.mean(var_list[size_bin*i:size_bin*(i+1)]) for i in range(num_bin)])
    t_auto = 0.5 * size_bin * np.mean(variance_binned) / np.mean(variance)
    stderr = abs(np.sqrt(variance_binned / num_bin))
    print('Energy = %.4f +- %.4f, Auto correlation Time = %.4f'%(mean, stderr, t_auto))
    return mean, stderr
