import tensorflow as tf

def get_shape(t):
    return t.get_shape().as_list()

def vars_info_vl(var_list): 
    return "    "+"\n    ".join(["{} : {}".format(v.name, get_shape(v)) for v in var_list])

def vars_info(string):
    return "Collection name %s\n"%string+vars_info_vl(tf.get_collection(string))

def get_init_vars(sess):
    init_vars = []
    for var in tf.global_variables():
        try: sess.run(var)
        except tf.errors.FailedPreconditionError: continue
        init_vars.append(var)
    return init_vars

def get_uninit_vars(sess):
    uninit_vars = []
    for var in tf.global_variables():
        try : sess.run(var)
        except tf.errors.FailedPreconditionError: uninit_vars.append(var)
    return uninit_vars

