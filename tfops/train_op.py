import tensorflow as tf

def get_train_op_v2(optimizer, loss, var_list=tf.trainable_variables()):
    return optimizer.minimize(loss=loss, var_list=var_list)
