import pdb
import numpy as np
import tensorflow as tf

from utils import typed_normal, sample_prob, save_img44

class RBM(object):
    def __init__(self, visible_input, num_hidden, eta=0.01):
        num_visible = visible_input.get_shape()[1].value
        dtype = visible_input.dtype

        self.weights = tf.Variable(typed_normal(dtype, (num_visible, num_hidden), eta), name='weights', dtype=dtype)
        self.hidden_bias = tf.Variable(typed_normal(dtype, (num_hidden,), eta), name='hidden-bias', dtype=dtype)
        self.visible_bias = tf.Variable(typed_normal(dtype, (num_visible,), eta), name='visible-bias', dtype=dtype)
        self.visible_input = visible_input

    def build_graph(self, name, cd_k=1):
        if name == 'square-loss' or name == 'free-energy-loss':
            # contractive divergence
            p_v = self.visible_input
            for step in range(cd_k):
                p_h, h = self._build_v2h(p_v, step)
                p_v, v = self._build_h2v(p_h, step)
            if name == 'free-energy-loss':
                with tf.variable_scope('free-energy-loss'):
                    loss = self._build_free_energy(self.visible_input) - self._build_free_energy(v)
            else:
                with tf.variable_scope('mean-square-loss'):
                    loss = tf.sqrt(tf.reduce_mean(tf.square(self.visible_input - p_v)))
            return v, loss
        elif name == 'v-prob':
            with tf.variable_scope('linear'):
                linear_out = tf.matmul(self.visible_input, self.weights) + self.hidden_bias
            return tf.reduce_prod(2*tf.cosh(linear_out))*tf.exp(tf.matmul(self.visible_input, self.visible_bias[:,None],1))
        elif name == 'free-energy':
            return self._build_free_energy(self.visible_input)
        else:
            raise NotImplementedError()

    def _build_free_energy(self, visible_batch):
        vbias_term = tf.matmul(visible_batch,self.visible_bias[:,None])
        wx_b = tf.matmul(visible_batch, self.weights)+self.hidden_bias
        hidden_term = tf.reduce_sum(tf.nn.softplus(wx_b), axis=1)
        return -tf.reduce_mean(hidden_term + vbias_term)

    def _build_v2h(self, v, token):
        wx_b = tf.matmul(v, self.weights) + self.hidden_bias
        p_h = tf.sigmoid(wx_b)
        sample_h = sample_prob(p_h, 'sample-hidden-%s'%token)
        return p_h, sample_h

    def _build_h2v(self, h, token):
        wx_b_ = tf.matmul(h, tf.transpose(self.weights)) + self.visible_bias
        p_v = tf.sigmoid(wx_b_)
        sample_v = sample_prob(p_v, 'sample-visible-%s'%token)
        return p_v, sample_v

def train(rbm, dataset, max_iter=2000):
    with tf.device('/cpu:0'):
        generated_img, loss = rbm.build_graph('free-energy-loss')
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

    # Test trained model
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for step in range(max_iter):
            batch_xs, batch_ys = dataset.train.next_batch(100)
            #loss_value = sess.run(loss, feed_dict={rbm.visible_input:batch_xs})
            _, loss_value = sess.run([train_step, loss], feed_dict={rbm.visible_input:batch_xs})
            print('step %d, loss = %s'%(step, loss_value))

        img_raw = batch_xs[:16].reshape([16,28,28])
        img_gen = sess.run(generated_img, feed_dict={rbm.visible_input:batch_xs})
        img_gen = img_gen[:16].reshape([16,28,28])
        save_img44('raw.png', img_raw)
        save_img44('gen.png', img_gen)

def test_train():
    # load mnist dataset
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets('MNIST-data', one_hot=True)

    # load predefined neural network model for mnist
    tf.set_random_seed(10086)
    rbm = load_demo_rbm('mnist')

    # train RBM using mnist dataset
    train(rbm, mnist, max_iter=2000)

def save_tensorboard_graph(name):
    rbm = load_demo_rbm('mnist')
    pv = rbm.build_graph(name)
    with tf.Session() as sess:
        writer = tf.summary.FileWriter("graphs", sess.graph)
        writer.close()

def get_variable_by_name(name):
    weight_var = tf.get_collection(tf.GraphKeys.VARIABLES, name)[0]

def load_demo_rbm(modelname):
    if modelname=='mnist':
        dtype = 'float32'
        num_batch, num_visible, num_hidden = 100, 784, 500
    elif modelname=='spin-chain':
        dtype = 'complex128'
        num_batch, num_visible, num_hidden = 1, 16, 4
    else:
        raise ValueError('undefined model name %s'%modelname)
    rbm = RBM(tf.placeholder(dtype, (num_batch, num_visible), name='visible_input'), num_hidden)
    return rbm

if __name__ == '__main__':
    #save_tensorboard_graph('v-prob')
    #save_tensorboard_graph('square-loss')
    #save_tensorboard_graph('free-energy-loss')
    #save_tensorboard_graph('free-energy')
    test_train()
