import tensorflow as tf


def sigmoid_cross_entropy(labels, logits):
    return tf.reduce_sum(tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits), axis=0))

def sigmoid_cross_entropy_without_mean(labels, logits):
    ndim = len(labels.get_shape().as_list())
    return tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits), axis=[idx for idx in range(1, ndim)])

def vae_kl_cost(mean, stddev, epsilon=1e-8):
    return tf.reduce_sum(tf.reduce_mean(0.5 * (tf.square(mean) + tf.square(stddev) - 2.0 * tf.log(stddev + epsilon) - 1.0), axis=0))

def vae_kl_cost_weight(mean, stddev, weight, epsilon=1e-8):
    return tf.reduce_sum(tf.multiply(tf.reduce_mean(0.5 * (tf.square(mean) + tf.square(stddev) - 2.0 * tf.log(stddev + epsilon) - 1.0), axis=0), weight))

