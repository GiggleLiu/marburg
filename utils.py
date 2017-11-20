import numpy as np
import tensorflow as tf

__all__ = ['sample_prob', 'typed_normal']

def sample_prob(probs, name):
    with tf.variable_scope(name):
        return tf.nn.relu(tf.sign(probs - tf.cast(tf.random_uniform(probs.shape, 0,1), probs.dtype)))

def typed_normal(dtype, shape, scale):
    dtype = dtype.name
    if dtype in ['complex64', 'complex128']:
        arr = np.random.normal(0,scale,shape) + 1j*np.random.normal(0,scale,shape)
    else:
        arr = np.random.normal(0, scale, shape)
    return arr.astype(dtype)

