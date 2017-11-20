import pdb
import numpy as np
import tensorflow as tf

from utils import typed_normal, sample_prob

class RBM(object):
    def __init__(self, visible_input, num_hidden, eta=0.01):
        num_visible = visible_input.get_shape()[1].value
        dtype = visible_input.dtype

        self.weights = tf.Variable(typed_normal(dtype, (num_hidden, num_visible), eta), name='weights', dtype=dtype)
        self.hidden_bias = tf.Variable(typed_normal(dtype, (num_hidden,), eta), name='hidden-bias', dtype=dtype)
        self.visible_bias = tf.Variable(typed_normal(dtype, (num_visible,), eta), name='visible-bias', dtype=dtype)
        self.visible_input = visible_input

    def build_graph(self, name, cd_k=1):
        if name == 'CD-loss':
            # contractive divergence
            p_v = self.visible_input
            for step in range(cd_k):
                p_h, h = self._build_v2h(p_v, step)
                p_v, v = self._build_h2v(p_h, step)
            with tf.variable_scope('mean-square-loss'):
                loss = tf.sqrt(tf.reduce_mean(tf.square(self.visible_input - p_v)))
            return v, loss
        elif name == 'v_prob':
            with tf.variable_scope('Linear'):
                linear_out = tf.matmul(self.visible_input, tf.transpose(self.weights)) + self.hidden_bias
            return tf.reduce_prod(2*tf.cosh(linear_out))*tf.exp(tf.matmul(self.visible_input, tf.reshape(self.visible_bias,(-1,1))))
        else:
            raise NotImplementedError()

    def _build_v2h(self, v, token):
        p_h = tf.sigmoid(tf.matmul(v, tf.transpose(self.weights)) + self.hidden_bias)
        sample_h = sample_prob(p_h, 'sample-hidden-%s'%token)
        return p_h, sample_h

    def _build_h2v(self, h, token):
        p_v = tf.sigmoid(tf.matmul(h, self.weights) + self.visible_bias)
        sample_v = sample_prob(p_v, 'sample-visible-%s'%token)
        return p_v, sample_v

def train(rbm, dataset):
    with tf.device('/cpu:0'):
        generated_img, loss = rbm.build_graph('CD-loss')
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

    # Test trained model
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for step in range(2000):
            batch_xs, batch_ys = dataset.train.next_batch(100)
            #loss_value = sess.run(loss, feed_dict={rbm.visible_input:batch_xs})
            _, loss_value = sess.run([train_step, loss], feed_dict={rbm.visible_input:batch_xs})
            print('step %d, loss = %s'%(step, loss_value))

def test_train():
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets('MNIST-data', one_hot=True)

    rbm = load_demo_rbm('mnist')
    train(rbm, mnist)

def visualize_graph(name):
    rbm = load_demo_rbm('mnist')
    pv = rbm.build_graph(name)
    with tf.Session() as sess:
        writer = tf.summary.FileWriter("graphs", sess.graph)
        writer.close()

def run_demo():
    tf.set_random_seed(10086)
    rbm = load_demo_rbm('mnist')
    pv = rbm.p_visible()
    with tf.Session() as sess:
        # initialize Variables like weights and biases.
        sess.run(tf.global_variables_initializer())
        for _ in range(2000):
            batch_xs, batch_ys = dataset.train.next_batch(100)
            sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
        sess.run(pv, feed_dict={rbm.visible_input:np.random.choice([-1,1], num_visible)})
        pdb.set_trace()

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
    # run_demo()
    #visualize_graph('v_prob')
    #visualize_graph('CD-loss')
    test_train()
