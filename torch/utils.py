import numpy as np
import pdb
import tensorflow as tf
import matplotlib.pyplot as plt

__all__ = ['sample_prob', 'typed_normal', 'save_img44', 'convolution_pbc',
        'sample_from_prob', 'show_images']


def sample_prob(probs, name):
    '''give 1 with specific probability, 0 otherwise.'''
    with tf.variable_scope(name):
        return tf.nn.relu(tf.sign(probs - tf.cast(tf.random_uniform(tf.shape(probs), 0, 1), probs.dtype)))


def typed_normal(dtype, shape, scale):
    '''normal distribution with specific data type.'''
    dtype = dtype.name
    if dtype in ['complex64', 'complex128']:
        arr = np.random.normal(0, scale, shape) + 1j * \
            np.random.normal(0, scale, shape)
    else:
        arr = np.random.normal(0, scale, shape)
    return arr.astype(dtype)


def save_img44(filename, data):
    import matplotlib.pyplot as plt
    nrow, ncol = 4, 4
    gs = plt.GridSpec(nrow, ncol)
    fig = plt.figure(figsize=(6, 5))
    for i in range(nrow):
        for j in range(ncol):
            ax = plt.subplot(gs[i, j])
            ax.imshow(data[i * ncol + j], interpolation='none', cmap='Greys')
            ax.axis('equal')
            ax.axis('off')
    gs.tight_layout(fig, pad=0)
    plt.show()

def numdiff(layer, x, var, dy, delta):
    '''numerical differenciation.'''
    var_raveled = var.ravel()

    var_delta_list = []
    for ix in range(len(var_raveled)):
        var_raveled[ix] += delta/2.
        yplus = layer.forward(x)
        var_raveled[ix] -= delta
        yminus = layer.forward(x)
        var_delta = ((yplus - yminus)/delta*dy).sum()
        var_delta_list.append(var_delta)

        # restore changes
        var_raveled[ix] += delta/2.
    return np.array(var_delta_list)

def sanity_check(layer, x, args, delta=0.01, precision=1e-3):
    '''
    perform sanity check for a layer,
    raise an assertion error if failed to pass all sanity checks.

    Args:
        layer (obj): user defined neural network layer.
        x (ndarray): input array.
        args: additional arguments for forward function.
        delta: the strength of perturbation used in numdiff.
        precision: the required precision of gradient (usually introduced by numdiff).
    '''
    y = layer.forward(x, *args)
    dy = np.random.randn(y.shape)
    x_delta = layer.backward(dy)

    for var, var_delta in zip([x] + layer.parameters, [dx] + layer.parameters_deltas):
        x_delta_num = numdiff(layer, x, var, dy, delta)
        assert(np.all(abs(x_delta_num - var_delta) < precision))
