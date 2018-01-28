import numpy as np

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
