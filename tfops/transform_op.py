from utils.np_op import zero_padding2nmul
from utils.tqdm_op import tqdm_range
from tqdm import tqdm

import tensorflow as tf
import numpy as np

def apply_tf_op(inputs, session, input_gate, output_gate, batch_size, train_gate=None, print_option=True):
    inputs, ndata = zero_padding2nmul(inputs=inputs, mul=batch_size)
    nbatch = len(inputs)//batch_size

    outputs = list()

    feed_dict = dict()
    if train_gate is not None: feed_dict[train_gate] = False 

    if print_option:
        for b in tqdm_range(nbatch):
            feed_dict[input_gate]=inputs[b*batch_size:(b+1)*batch_size]
            outputs.append(session.run(output_gate, feed_dict=feed_dict))
    else:
        for b in range(nbatch):
            feed_dict[input_gate]=inputs[b*batch_size:(b+1)*batch_size]
            outputs.append(session.run(output_gate, feed_dict=feed_dict))

    outputs = np.concatenate(outputs, axis=0)
    outputs = outputs[:ndata]

    return outputs

def apply_tf_op_multi_output(inputs, session, input_gate, output_gate_list, batch_size, train_gate=None, print_option=True):
    inputs, ndata = zero_padding2nmul(inputs=inputs, mul=batch_size)
    nbatch = len(inputs)//batch_size

    noutput = len(output_gate_list)
    outputs_list = [list() for o_idx in range(noutput)]

    feed_dict = dict()
    if train_gate is not None: feed_dict[train_gate] = False 

    if print_option:
        for b in tqdm_range(nbatch):
            feed_dict[input_gate]=inputs[b*batch_size:(b+1)*batch_size]
            tmp = session.run(output_gate_list, feed_dict=feed_dict)
            for o_idx in range(noutput):
                outputs_list[o_idx].append(tmp[o_idx])
    else:
        for b in range(nbatch):
            feed_dict[input_gate]=inputs[b*batch_size:(b+1)*batch_size]
            tmp = session.run(output_gate_list, feed_dict=feed_dict)
            for o_idx in range(noutput):
                outputs_list[o_idx].append(tmp[o_idx])

    for o_idx in range(noutput):
        outputs_list[o_idx] = np.concatenate(outputs_list[o_idx], axis=0)
        outputs_list[o_idx] = outputs_list[o_idx][:ndata]

    return outputs_list

def apply_tf_op_multi_input(inputs_list, session, input_gate_list, output_gate, batch_size, train_gate=None, print_option=True):
    assert len(inputs_list)==len(input_gate_list), "Length of list should be same"
    ninput = len(inputs_list)
    inputs_pad_list = list()
    ndata = len(inputs_list[0])

    for i_idx in range(ninput):
        inputs_pad_list.append(zero_padding2nmul(inputs=inputs_list[i_idx], mul=batch_size)[0])
        
    nbatch = len(inputs_pad_list[0])//batch_size

    outputs = list()
    feed_dict = dict()
    if train_gate is not None: feed_dict[train_gate] = False 

    if print_option:
        for b in tqdm_range(nbatch):
            for i_idx in range(ninput): feed_dict[input_gate_list[i_idx]]=inputs_pad_list[i_idx][b*batch_size:(b+1)*batch_size]
            outputs.append(session.run(output_gate, feed_dict=feed_dict))
    else:
        for b in range(nbatch):
            for i_idx in range(ninput): feed_dict[input_gate_list[i_idx]]=inputs_pad_list[i_idx][b*batch_size:(b+1)*batch_size]
            outputs.append(session.run(output_gate, feed_dict=feed_dict))

    outputs = np.concatenate(outputs, axis=0)
    outputs = outputs[:ndata]
    return outputs

