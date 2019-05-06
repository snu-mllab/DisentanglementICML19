import os
import sys
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..'))

from utils.tqdm_op import tqdm_range

import numpy as np
import copy

def np_softmax(x):
    '''
    Args:
        x - Numpy 2D array
    '''
    x_softmax = np.zeros_like(x)

    ndata, nfeature = x.shape

    for idx in range(ndata):
        tmp_max = np.max(x[idx])
        tmp_exp = np.exp(x[idx] - tmp_max)
        x_softmax[idx] = tmp_exp/np.sum(tmp_exp)
    return x_softmax

def get_ginni_variance_conti(array):
    ''' FactorVAE https://arxiv.org/pdf/1802.05983.pdf
    Args:
        array - Numpy 1D array
    '''
    ndata = array.shape[0]
    return ndata/(ndata-1)*np.var(array)

def get_ginni_variance_discrete(array):
    ''' FactorVAE https://arxiv.org/pdf/1802.05983.pdf
    Args: array - Numpy 1D array, argmax index
    '''
    array = array.astype(int)
    ndata = array.shape[0]
    count = np.zeros([np.max(array)+1])
    for idx in range(ndata): count[array[idx]]+=1
    count = count.astype(float)
    return (ndata*ndata - np.sum(np.square(count)))/(2*ndata*(ndata-1))

def zero_padding2nmul(inputs, mul):
    '''Add zero padding to inputs to be multiple of mul
    Args:
        inputs - np array
        mul - int

    Return:
        np array (inputs + zero_padding)
        int original input size
    '''
    input_shape = list(inputs.shape)
    ndata = input_shape[0]
    if ndata%mul==0: return inputs, ndata
    input_shape[0] = mul-ndata%mul
    return np.concatenate([inputs, np.zeros(input_shape)], axis=0), ndata 

def np_random_crop_4d(imgs, size):
    '''
    Args:
        imgs - 4d image NHWC
        size - list (rh, rw)
    '''
    rh, rw = size
    on, oh, ow, oc = imgs.shape

    cropped_imgs = np.zeros([on, rh, rw, oc])
    ch = np.random.randint(low=0, high=oh-rh, size=on)
    cw = np.random.randint(low=0, high=ow-rw, size=on)
    for idx in range(on): cropped_imgs[idx] = imgs[idx,ch[idx]:ch[idx]+rh,cw[idx]:cw[idx]+rw]
    return cropped_imgs
