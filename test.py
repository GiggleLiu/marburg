import numpy as np
from numpy.testing import dec, assert_, assert_raises,\
    assert_almost_equal, assert_allclose

from vmc import heisenberg_loc, heisenberg_loc_tf

def test_heisenberg_loc():
    J = 0.2
    config = np.array([1,0,0,1,1.])
    wave_func = lambda x: 1.0
    eloc1 = heisenberg_loc(J, config, wave_func)
    eloc2 = heisenberg_loc_tf(J, config, wave_func)
