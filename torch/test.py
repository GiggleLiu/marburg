from __future__ import division
import numpy as np
import pdb
import tensorflow as tf
from numpy.testing import dec, assert_, assert_raises,\
    assert_almost_equal, assert_allclose

from vmc import vmc_measure
from hamiltonians import heisenberg_loc
from train import VMCKernel
from rbm_torch import RBM
from train import train


class FakeModel(object):
    def __init__(self, wave_func='exact'):
        self.J = 1.0
        self.nsite = 8
        H, self.E_exact, self.wf = get_heisenberg_ground_state(
            self.J, self.nsite)
        self._configs = np.unpackbits(
            np.arange(2**self.nsite)[:, None].astype('uint8'), axis=1)
        self._mask = self._configs.sum(
            axis=1) == self.nsite // 2  # Sz = 0 block
        if wave_func == 'random' or not isinstance(wave_func, str):
            if wave_func == 'random':
                psi = np.random.random(2**self.nsite)
            else:
                psi = wave_func
            psi[~self._mask] = 0
            self.wf = psi / np.linalg.norm(psi)
            self.E_exact = self.wf.dot(H.dot(self.wf))

        assert_allclose([self.get_wf(config)
                         for config in self._configs], self.wf)

    def get_wf(self, config):
        # 0 = spin up, 1 = spin down
        return self.wf[np.packbits(config).item()]

    def prob(self, config):
        return np.abs(self.get_wf(config))**2

    def next_batch(self, batch_size):
        '''sample wave function'''
        p = np.abs(self.wf)**2
        p = p / p.sum()
        indices = np.random.choice(np.arange(len(self.wf)), batch_size, p=p)
        res = np.unpackbits(indices[:, None].astype('uint8'), axis=1)
        return res, None

    def local_measure(self, config):
        '''get E_loc, grad_loc.'''
        # {d/dW}_{loc}
        grad_locs = np.array([0])

        # E_{loc}
        E_loc = heisenberg_loc(config, self.get_wf, psi_loc=self.get_wf(config))
        return E_loc, grad_locs

    @staticmethod
    def propose_config(old_config, prob_flip=0.05):
        '''
        flip two positions as suggested spin flips.
        '''
        # take ~ 5% probability to flip all spin, can making VMC sample better in Heisenberg model
        if np.random.random() < prob_flip:
            return 1-old_config

        num_spin = len(old_config)
        upmask = old_config == 0
        flips = np.random.randint(0, num_spin // 2, 2)
        iflip0 = np.where(upmask)[0][flips[0]]
        iflip1 = np.where(~upmask)[0][flips[1]]

        config = old_config.copy()
        config[iflip0] = 1
        config[iflip1] = 0
        return config

def get_heisenberg_ground_state(J, nsite):
    '''Get the target Hamiltonian Matrix.'''
    from scipy.sparse import kron, eye, csr_matrix
    from scipy.sparse.linalg import eigsh
    sx = csr_matrix([[0, 1], [1, 0]])
    sy = csr_matrix([[0, -1j], [1j, 0]])
    sz = csr_matrix([[1, 0], [0, -1]])

    h2 = -J / 4. * (kron(sx, sx) + kron(sy, sy)) + J / 4. * kron(sz, sz)
    H = 0
    for i in range(nsite - 1):
        H = H + kron(kron(eye(2**i), h2), eye(2**(nsite - 2 - i)))

    # impose periodic boundary
    H = H - J / 4. * (kron(kron(sx, eye(2**(nsite - 2))), sx) + kron(kron(sy, eye(
        2**(nsite - 2))), sy)) + J / 4. * (kron(kron(sz, eye(2**(nsite - 2))), sz))
    #E, V = eigsh(H, which='SA', k=1)
    #return H, E.item(), V.ravel()
    E, V = np.linalg.eigh(H.toarray())
    return H, E[0], V[:,0]


def test_train_rbm_using_exact_wf():
    dtype = 'float32'
    model = FakeModel()
    num_batch, num_visible, num_hidden = None, model.nsite, 4
    rbm = RBM(num_visible, num_hidden)

    # train RBM on model
    psi = rbm.build_graph('psi')

    def project_v(v):
        v = np.sqrt(v)
        #v[~model._mask] = 0
        return v / np.linalg.norm(v)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        v_ = sess.run(psi, feed_dict={rbm.visible_input: model._configs})
        v = project_v(v_)
        print('overlap = %s' % np.abs(v.T.dot(model.wf)**2))

        graph, loss = train(rbm, model.next_batch, sess,
                            max_iter=10000, learning_rate=0.01, cd_k=1, batch_size=500)

        v_ = sess.run(psi, feed_dict={rbm.visible_input: model._configs})
        v = project_v(v_)
        print('overlap = %s' % np.abs(v.T.dot(model.wf)**2))


def test_vmc_random():
    np.random.seed(5)
    print('Testing local energy, VMC and config proposal.')
    model = FakeModel(wave_func='random')
    E_mean, grad_mean, Egrad_mean, E_err = vmc_measure(
        model, initial_config=np.array([0, 1] * (model.nsite // 2)), num_bath=2000, num_sample=10000)
    print('Get energy %.6f (exact = %.6f)' % (E_mean, model.E_exact))
    assert_almost_equal(E_mean, model.E_exact, decimal=1)


def test_vmc_rbm():
    print('Testing local energy, VMC and config proposal.')
    E_exact = 3.65109341
    num_visible, num_hidden = 8, 4

    # true model
    rbm = RBM(num_visible, num_hidden)
    model_ = VMCKernel(ansatz=rbm, energy_loc = heisenberg_loc)
    # wave function
    model = FakeModel()
    wf = np.array([model_.psi(c).data.numpy().item() for c in model._configs])
    model = FakeModel(wave_func=wf)

    E_mean, grad_mean, Egrad_mean, E_err = vmc_measure(
        model, initial_config=np.array([0, 1] * (model.nsite // 2)), num_bath=200, num_sample=10000)
    E_mean_, grad_mean_, Egrad_mean_, E_err_ = vmc_measure(
        model_, initial_config=np.array([0, 1] * (model.nsite // 2)), num_bath=200, num_sample=10000)
    print('Get energy %.6f (exact = %.6f)' % (E_mean.real, model.E_exact.real))
    print('True model Get energy %.6f (exact = %.6f)' %
          (E_mean_.real, model.E_exact.real))
    assert_almost_equal(E_mean, model.E_exact, decimal=1)

def test_binary():
    rbm = BinaryRBM(4, 4)
    pv = rbm.pv(Variable(torch.Tensor([-1, 1, 1, -1])))
    ph = rbm.ph(Variable(torch.Tensor([-1, 1, 1, -1])))
    print(pv, ph)
    pdb.set_trace()


if __name__ == '__main__':
    np.random.seed(2)
    # test_vmc_random()
    test_vmc_rbm()
    #test_train_rbm_using_exact_wf()
