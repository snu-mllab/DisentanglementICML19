import tensorflow as tf

def stair_decay(initial_lr, decay_steps, decay_rate, initial_step=0):
    #with tf.variable_scope(scope): global_step = tf.Variable(initial_step, name='global_step', trainable=False)
    global_step = tf.Variable(initial_step, trainable=False)
    update_step_op = tf.assign_add(global_step, 1)
    return tf.train.exponential_decay(
                learning_rate=initial_lr,\
                global_step=global_step,\
                decay_steps=decay_steps,\
                decay_rate=decay_rate,\
                staircase=True), update_step_op

def piecewise_decay(boundaries, values, initial_step = 0):
    #with tf.variable_scope(scope): global_step = tf.Variable(initial_step, name='global_step', trainable=False)
    global_step = tf.Variable(initial_step, name='global_step', trainable=False)
    update_step_op = tf.assign_add(global_step, 1)
    return tf.train.piecewise_constant(global_step, boundaries, values), update_step_op

DECAY_DICT = {
            'stair' : stair_decay,
            'piecewise' : piecewise_decay
            }

DECAY_PARAMS_DICT =\
    {
    'stair' : 
        {
            64 :{
                'a1' : {'initial_lr' : 1e-5, 'decay_steps' : 50000, 'decay_rate' : 0.3},
                'a2' : {'initial_lr' : 3e-5, 'decay_steps' : 50000, 'decay_rate' : 0.3},
                'a3' : {'initial_lr' : 1e-4, 'decay_steps' : 50000, 'decay_rate' : 0.3},
                'a4' : {'initial_lr' : 3e-4, 'decay_steps' : 50000, 'decay_rate' : 0.3},
                'a5' : {'initial_lr' : 1e-3, 'decay_steps' : 50000, 'decay_rate' : 0.3}
                }
        }
    }

