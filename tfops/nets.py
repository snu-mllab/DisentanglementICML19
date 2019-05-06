import os
import sys
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..'))

import tensorflow as tf

slim = tf.contrib.slim
#=============================================================================================================================================#
def encoder1_32(x, output_dim, output_nonlinearity=None, scope="ENC", reuse=False):
    nets_dict = dict()
    nets_dict['input'] = x
    with tf.variable_scope(scope, reuse=reuse):
        with slim.arg_scope([slim.conv2d, slim.fully_connected], weights_regularizer=slim.l2_regularizer(0.00004)):
            with slim.arg_scope([slim.conv2d], weights_initializer=tf.contrib.slim.variance_scaling_initializer(), stride=2, padding='SAME', activation_fn=tf.nn.relu) :
                with slim.arg_scope([slim.fully_connected], biases_initializer=tf.zeros_initializer()):
                    nets_dict['conv2d0'] = slim.conv2d(nets_dict['input'], 32, [4, 4], scope='conv2d_0')
                    nets_dict['conv2d1'] = slim.conv2d(nets_dict['conv2d0'], 32, [4, 4], scope='conv2d_1')
                    nets_dict['conv2d2'] = slim.conv2d(nets_dict['conv2d1'], 64, [4, 4], scope='conv2d_2')
                    n = tf.reshape(nets_dict['conv2d2'], [-1, 4*4*64])
                    nets_dict['fc0'] = slim.fully_connected(n, 256, activation_fn=tf.nn.relu, scope = "output_fc0")
                    nets_dict['output'] = slim.fully_connected(nets_dict['fc0'], output_dim, activation_fn=output_nonlinearity, scope = "output_fc1")
                    return nets_dict
def decoder1_32(z, scope="DEC", reuse=False):
    nets_dict = dict()
    nets_dict['input'] = z
    with tf.variable_scope(scope, reuse=reuse):
        with slim.arg_scope([slim.conv2d_transpose, slim.fully_connected], weights_regularizer=slim.l2_regularizer(0.00004)):
            with slim.arg_scope([slim.conv2d_transpose], weights_initializer=tf.contrib.slim.variance_scaling_initializer(),
                    stride=2, padding='SAME', activation_fn=tf.nn.relu): 
                with slim.arg_scope([slim.fully_connected], biases_initializer=tf.zeros_initializer()):
                    nets_dict['fc0'] = slim.fully_connected(nets_dict['input'], 256, activation_fn=tf.nn.relu, scope = "fc0")
                    nets_dict['fc1'] = slim.fully_connected(nets_dict['fc0'], 4*4*64, activation_fn=tf.nn.relu, scope = "fc1")
                    n = tf.reshape(nets_dict['fc1'], [-1, 4, 4, 64])
                    nets_dict['deconv2d0'] = slim.conv2d_transpose(n, 32, [4, 4], scope='deconv2d_0')
                    nets_dict['deconv2d1'] = slim.conv2d_transpose(nets_dict['deconv2d0'], 32, [4, 4], scope='deconv2d_1')
                    nets_dict['output'] = slim.conv2d_transpose(nets_dict['deconv2d1'], 1, [4, 4], activation_fn=None, scope='deconv2d_2')
                    return nets_dict
#=============================================================================================================================================#
def encoder1_64(x, output_dim, output_nonlinearity=None, scope="ENC", reuse=False):
    nets_dict = dict()
    nets_dict['input'] = x
    with tf.variable_scope(scope, reuse=reuse):
        with slim.arg_scope([slim.conv2d, slim.fully_connected], weights_regularizer=slim.l2_regularizer(0.00004)):
            with slim.arg_scope([slim.conv2d], weights_initializer=tf.contrib.slim.variance_scaling_initializer(), stride=2, padding='SAME', activation_fn=tf.nn.relu) :
                with slim.arg_scope([slim.fully_connected], biases_initializer=tf.zeros_initializer()):
                    nets_dict['conv2d0'] = slim.conv2d(nets_dict['input'], 32, [4, 4], scope='conv2d_0')
                    nets_dict['conv2d1'] = slim.conv2d(nets_dict['conv2d0'], 32, [4, 4], scope='conv2d_1')
                    nets_dict['conv2d2'] = slim.conv2d(nets_dict['conv2d1'], 64, [4, 4], scope='conv2d_2')
                    nets_dict['conv2d3'] = slim.conv2d(nets_dict['conv2d2'], 64, [4, 4], scope='conv2d_3')
                    n = tf.reshape(nets_dict['conv2d3'], [-1, 4*4*64])
                    nets_dict['fc0'] = slim.fully_connected(n, 256, activation_fn=tf.nn.relu, scope = "output_fc0")
                    nets_dict['output'] = slim.fully_connected(nets_dict['fc0'], output_dim, activation_fn=output_nonlinearity, scope = "output_fc1")
                    return nets_dict

def decoder1_64(z, scope="DEC", output_channel=1, reuse=False):
    nets_dict = dict()
    nets_dict['input'] = z
    with tf.variable_scope(scope, reuse=reuse):
        with slim.arg_scope([slim.conv2d_transpose, slim.fully_connected], weights_regularizer=slim.l2_regularizer(0.00004)):
            with slim.arg_scope([slim.conv2d_transpose], weights_initializer=tf.contrib.slim.variance_scaling_initializer(),
                    stride=2, padding='SAME', activation_fn=tf.nn.relu): 
                with slim.arg_scope([slim.fully_connected], biases_initializer=tf.zeros_initializer()):
                    nets_dict['fc0'] = slim.fully_connected(nets_dict['input'], 256, activation_fn=tf.nn.relu, scope = "fc0")
                    nets_dict['fc1'] = slim.fully_connected(nets_dict['fc0'], 4*4*64, activation_fn=tf.nn.relu, scope = "fc1")
                    n = tf.reshape(nets_dict['fc1'], [-1, 4, 4, 64])
                    nets_dict['deconv2d0'] = slim.conv2d_transpose(n, 64, [4, 4], scope='deconv2d_0')
                    nets_dict['deconv2d1'] = slim.conv2d_transpose(nets_dict['deconv2d0'], 32, [4, 4], scope='deconv2d_1')
                    nets_dict['deconv2d2'] = slim.conv2d_transpose(nets_dict['deconv2d1'], 32, [4, 4], scope='deconv2d_2')
                    nets_dict['output'] = slim.conv2d_transpose(nets_dict['deconv2d2'], output_channel, [4, 4], activation_fn=None, scope='deconv2d_3')
                    return nets_dict

