import pdb
import numpy as np
import tensorflow as tf

from utils import typed_normal, sample_prob, save_img44, convolution_pbc


class RBM(object):
    def __init__(self, visible_input, num_hidden, eta=0.1):
        num_visible = visible_input.get_shape()[1].value
        dtype = visible_input.dtype

        self.weights = tf.Variable(typed_normal(
            dtype, (num_visible, num_hidden), eta), name='weights', dtype=dtype)
        self.hidden_bias = tf.Variable(typed_normal(
            dtype, (num_hidden,), eta), name='hidden-bias', dtype=dtype)
        self.visible_bias = tf.Variable(typed_normal(
            dtype, (num_visible,), eta), name='visible-bias', dtype=dtype)
        self.visible_input = visible_input

    def build_graph(self, name, cd_k=1):
        '''
        build a graph

        Args:
            name (str): specify computational graph name.
                * free-energy-loss: get the free energy loss F(v_data)-F(v_model).
                * psi: get the probability distribution from visual node.
            cd_k (int): number of forward-backward iteration in '-loss' graphs.
        '''
        if name == 'square-loss' or name == 'free-energy-loss':
            # contractive divergence
            p_v = self.visible_input
            for step in range(cd_k):
                with tf.variable_scope('gibbs-sampling-%d' % (step + 1)):
                    p_h, h = self._build_v2h(p_v, step)
                    p_v, v = self._build_h2v(p_h, step)
            if name == 'free-energy-loss':
                with tf.variable_scope('free-energy-loss'):
                    loss = self._build_free_energy(
                        self.visible_input) - self._build_free_energy(v)
            else:
                with tf.variable_scope('mean-square-loss'):
                    loss = tf.sqrt(tf.reduce_mean(
                        tf.square(self.visible_input - p_v)))
            return v, loss
        elif name == 'psi':
            with tf.variable_scope('linear'):
                linear_out = tf.matmul(
                    self.visible_input, self.weights) + self.hidden_bias
            #return tf.reduce_prod(2 * tf.cosh(linear_out), axis=1) * tf.exp(tf.matmul(self.visible_input, self.visible_bias[:, None]))[:, 0]
            return tf.exp(tf.reduce_sum(tf.nn.softplus(linear_out), axis=(1,)))
        elif name == 'psi-cnn':
            with tf.variable_scope('colvolution-pbc'):
                # input shape = [batch_size] + input_spatial_shape + [in_channels]
                # filter shape = spatial_filter_shape + [in_channels, out_channels]
                # output shape = [batch_size] +
                # here we have in_channels == 1.
                linear_out = convolution_pbc(
                    self.visible_input[:, :, None], self.weights[:, None, :]) + self.hidden_bias
            #return tf.exp(tf.reduce_sum(tf.log(tf.cosh(linear_out)), axis=(1, 2)))
            return tf.exp(tf.reduce_sum(tf.nn.softplus(linear_out), axis=(1, 2)))
        elif name == 'free-energy':
            return self._build_free_energy(self.visible_input)
        else:
            raise NotImplementedError()

    def _build_free_energy(self, visible_batch):
        vbias_term = tf.matmul(visible_batch, self.visible_bias[:, None])
        wx_b = tf.matmul(visible_batch, self.weights) + self.hidden_bias
        hidden_term = tf.reduce_sum(tf.nn.softplus(wx_b), axis=1)
        return -tf.reduce_mean(hidden_term + vbias_term)

    def _build_v2h(self, v, token):
        wx_b = tf.matmul(v, self.weights) + self.hidden_bias
        p_h = tf.sigmoid(wx_b)
        sample_h = sample_prob(p_h, 'sample-hidden-%s' % token)
        return p_h, sample_h

    def _build_h2v(self, h, token):
        wx_b_ = tf.matmul(h, tf.transpose(self.weights)) + self.visible_bias
        p_v = tf.sigmoid(wx_b_)
        sample_v = sample_prob(p_v, 'sample-visible-%s' % token)
        return p_v, sample_v


def train(rbm, batch_generator, sess, batch_size=100,
        max_iter=2000, learning_rate=0.1, cd_k=1):
    #generated_img, loss = rbm.build_graph('free-energy-loss', cd_k=cd_k)
    generated_img, loss = rbm.build_graph('square-loss', cd_k=cd_k)
    train_step = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)
    sess.run(tf.global_variables_initializer())

    # Test trained model
    for step in range(max_iter):
        batch_xs, batch_ys = batch_generator(batch_size)
        _, loss_value = sess.run([train_step, loss], feed_dict={
                                 rbm.visible_input: batch_xs})
        print('step %d, loss = %s' % (step, loss_value))
    return generated_img, loss


def test_train():
    # load mnist dataset
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets('MNIST-data', one_hot=True)

    # load predefined neural network model for mnist
    tf.set_random_seed(10086)
    rbm = load_demo_rbm('mnist')

    # Test set
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # train RBM using mnist dataset
        generated_img, loss = train(
            rbm, mnist.train.next_batch, sess, max_iter=2000)

        batch_xs, batch_ys = mnist.test.next_batch(16)
        loss_value = sess.run(loss, feed_dict={
            rbm.visible_input: batch_xs})
        print('loss = %s' % (loss_value))

        img_raw = batch_xs.reshape([16, 28, 28])
        img_gen = sess.run(generated_img, feed_dict={
                           rbm.visible_input: batch_xs})
        img_gen = img_gen.reshape([16, 28, 28])
        save_img44('data/images/raw.png', img_raw)
        save_img44('data/images/gen.png', img_gen)


def save_tensorboard_graph(name, cd_k=1):
    rbm = load_demo_rbm('mnist')
    pv = rbm.build_graph(name, cd_k=cd_k)
    with tf.Session() as sess:
        writer = tf.summary.FileWriter("graphs", sess.graph)
        writer.close()


def load_demo_rbm(modelname):
    if modelname == 'mnist':
        dtype = 'float32'
        num_batch, num_visible, num_hidden = None, 784, 500
    elif modelname == 'spin-chain':
        dtype = 'float32'
        num_batch, num_visible, num_hidden = 1, 8, 4
    else:
        raise ValueError('undefined model name %s' % modelname)
    rbm = RBM(tf.placeholder(dtype, (num_batch, num_visible),
                             name='visible_input'), num_hidden)
    return rbm


if __name__ == '__main__':
    save_tensorboard_graph('psi')
    # save_tensorboard_graph('square-loss')
    # save_tensorboard_graph('free-energy-loss', 2)
    # save_tensorboard_graph('free-energy')
    # test_train()
