from __future__ import division
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
        eng_i = wi * psi_func(config_i) / psi_loc
        acc += eng_i
    return acc

def J1J2_loc(config, psi_func, psi_loc, J1, J1z, J2, J2z, periodic):
    '''
    local energy for 1D J1-J2 chain.

    Args:
        config (1darray): bit string as spin configuration.
        psi_func (func): wave function.
        psi_loc (number): wave function projected on configuration <config|psi>.
        J1, J1z, J2, J2z (float): coupling strenghs for nearest neightbor, nearest neighbor z-direction, 2nd nearest neighbor and 2nd nearest neighbor in z-direction.
        periodic (bool): boundary condition.

    Returns:
        number: local energy.
    '''
    nsite = len(config)
    wl, flips = [], []
    Js = [J1, J2]
    Jzs = [J1z, J2z]
    for INB, (J,Jz) in enumerate(zip(Js, Jzs)):
        # J1(SzSz) terms.
        nn_par = np.roll(config, -INB-1) * config
        if not periodic:
            nn_par = nn_par[:-INB-1]
        wl.append(Jz / 4. * (nn_par).sum(axis=-1))
        flips.append(np.array([], dtype='int64'))

        # J1(SxSx) and J1(SySy) terms.
        mask = nn_par != 1
        i = np.where(mask)[0]
        j = (i + INB+1) % nsite

        wl += [J / 2.] * len(i)
        flips.extend(zip(i, j))
        
    # calculate local energy <psi|H|sigma>/<psi|sigma>
    acc = 0
    for wi, flip in zip(wl, flips):
        config_i = config.copy()
        config_i[list(flip)] *= -1
        eng_i = wi * psi_func(config_i) / psi_loc
        acc += eng_i
    return acc


def J1J2_2D_loc(config, psi_func, psi_loc, J1, J1z, J2, J2z, N1, N2, periodic):
    '''
    local energy for 2D J1-J2 chain.

    Args:
        config (1darray): bit string as spin configuration.
        psi_func (func): wave function.
        psi_loc (number): wave function projected on configuration <config|psi>.
        J1, J1z, J2, J2z (float): coupling strenghs for nearest neightbor, nearest neighbor z-direction, 2nd nearest neighbor and 2nd nearest neighbor in z-direction.
        N1, N2 (int): lattice size in x, y directions.
        periodic (bool): boundary condition.

    Returns:
        number: local energy.
    '''
    nsite = N1 * N2
    Js=[J1, J2]
    Jzs=[J1z, J2z]
    config2d = config.reshape([N1, N2])
    wl, flips = [], []

    for INB, (J,Jz) in enumerate(zip(Js, Jzs)):
        # J(SzSz) terms.
        if INB==0:
            nn_par1 = np.roll(config2d, -1, axis=0) * config2d
            nn_par2 = np.roll(config2d, -1, axis=1) * config2d
            if not periodic:
                nn_par1 = nn_par1[:-1, :]
                nn_par2 = nn_par2[:, :-1]
        else:
            nn_par1 = np.roll(np.roll(config2d, -1, axis=0), -1, axis=1) * config2d
            nn_par2 = np.roll(np.roll(config2d, -1, axis=1), 1, axis=0) * config2d
            nn_par2_ = np.roll(np.roll(np.arange(nsite).reshape([N1,N2]), -1, axis=1), 1, axis=0) * config2d
            if not periodic:
                nn_par1 = nn_par1[:-1, :-1]
                nn_par2 = nn_par2[1:, :-1]
                nn_par2_ = nn_par2_[1:,:-1]
        wl.append(Jz / 4. * (nn_par1.sum() + nn_par2.sum()))
        flips.append(np.array([], dtype='int64'))
        # return wl,flips

        # bond in 1st direction
        mask1 = nn_par1 != 1
        i1, i2 = np.where(mask1)
        j1, j2 = (i1 + 1) % N1, (i2 if INB==0 else (i2 + 1)) % N2
        i = i1 * N2 + i2
        j = j1 * N2 + j2
        wl += [J / 2.] * len(i)
        flips.extend(zip(i, j))

        # bond in 2nd direction
        mask2 = nn_par2 != 1
        ix, iy = np.where(mask2)
        if INB==1 and not periodic:
            ix = (ix+1)%N1
        jx, jy = (ix if INB==0 else (ix - 1)) % N1, (iy + 1) % N2
        i = ix * N2 + iy
        j = jx * N2 + jy
        wl += [J / 2.] * len(i)
        flips.extend(zip(i, j))

    # calculate local energy <psi|H|sigma>/<psi|sigma>
    acc = 0
    for wi, flip in zip(wl, flips):
        config_i = config.copy()
        config_i[list(flip)] *= -1
        eng_i = wi * psi_func(config_i) / psi_loc
        acc += eng_i
    return acc
