import numpy as np

def heisenberg_loc(config, psi_func, psi_loc, J=1.):
    '''
    local energy for 1D Periodic Heisenberg chain.

    Args:
        config (1darray): bit string as spin configuration.
        psi_func (func): wave function.
        psi_loc (number): wave function projected on configuration <config|psi>.
        J (float): coupling strengh.

    Returns:
        number: local energy.
    '''
    # get weights and flips after applying hamiltonian \sum_i w_i|sigma_i> = H|sigma>
    nsite = len(config)
    wl, flips = [], []
    # J*SzSz terms.
    nn_par = 1 - 2. * (np.roll(config, -1) ^ config)
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
        config_i[list(flip)] = 1 - config_i[list(flip)]
        eng_i = wi * psi_func(config_i) / psi_loc
        acc += eng_i
    return acc
