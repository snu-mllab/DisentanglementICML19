import os
import sys
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..'))

from utils.np_op import get_ginni_variance_discrete, get_ginni_variance_conti

import numpy as np

def DisentanglemetricFactorMask(mean, std, nclasses, sampler, nvote=800, nex_per_vote=100, eps=1e-10, thr=0.1, print_option=False): 
    kl_cost = np.mean(0.5*(np.square(mean)+np.square(std)-1-2*np.log(std+eps)),axis=0)
    return DisentanglemetricFactorMaskcustom(latent_conti=mean, mi_array=kl_cost, nclasses=nclasses, sampler=sampler, nvote=nvote, nex_per_vote=nex_per_vote, eps=eps, thr=thr, print_option=print_option)

def DisentanglemetricFactorJointMask(mean, std, latent_cat, nclasses, sampler, nvote=800, nex_per_vote=100, eps=1e-10, thr=0.1, print_option=False, ignore_discrete=True): 
    return DisentanglemetricFactorMultiJointMask(mean=mean, std=std, latent_cat_list=[latent_cat], nclasses=nclasses, sampler=sampler, nvote=nvote, nex_per_vote=nex_per_vote, eps=eps, thr=thr, print_option=print_option, ignore_discrete=ignore_discrete)

def DisentanglemetricFactorMultiJointMask(mean, std, latent_cat_list, nclasses, sampler, nvote=800, nex_per_vote=100, eps=1e-10, thr=0.1, print_option=False, ignore_discrete=True): 
    kl_cost = np.mean(0.5*(np.square(mean)+np.square(std)-1-2*np.log(std+eps)),axis=0)

    tmp = list()
    for latent_cat in latent_cat_list:
        if ignore_discrete:
            kl_cat_cost = np.log(latent_cat.shape[1]) + np.mean(np.sum(latent_cat*np.log(latent_cat+eps), axis=1))
            if kl_cat_cost < thr:
                tmp.append(np.argmax(latent_cat, axis=-1))
        else:
            tmp.append(np.argmax(latent_cat, axis=-1))

    if len(tmp)==0:
        print("Ignore discrete variable")
        return DisentanglemetricFactorMaskcustom(latent_conti=mean, mi_array=kl_cost, nclasses=nclasses, sampler=sampler, nvote=nvote, nex_per_vote=nex_per_vote, eps=eps, thr=thr, print_option=print_option)
    else:
        return DisentanglemetricFactorMultiJointMaskcustom(latent_conti=mean, latent_cat_list=tmp, mi_array=kl_cost, nclasses=nclasses, sampler=sampler, nvote=nvote, nex_per_vote=nex_per_vote, eps=eps, thr=thr, print_option=print_option)

def DisentanglemetricFactorMaskcustom(latent_conti, mi_array, nclasses, sampler, nvote=800, nex_per_vote=100, eps=1e-10, thr=0.1, print_option=False): 
    mask = np.where(mi_array>thr)[0]
    latent_conti = latent_conti[:, mask]
    if print_option:
        print(mask)

    k_set = list()
    for idx in range(nclasses.shape[0]):
        if nclasses[idx]>1: k_set.append(idx)
    nfactor = len(k_set)

    if print_option:
        print(k_set)

    if latent_conti.shape[1]==0: return 1/nfactor

    var = np.array([get_ginni_variance_conti(latent_conti[:, v]) for v in range(latent_conti.shape[1])])

    nlatent = var.shape[0]
    nvote_per_factor = int(nvote/nfactor)
    
    count = np.zeros([nlatent, nfactor])
    for idx in range(nfactor):
        for iter_ in range(nvote_per_factor):
            k_fixed = k_set[idx]
            fixed_value = np.random.randint(nclasses[k_fixed])
            batch_latent_conti = latent_conti[sampler(batch_size=nex_per_vote, latent_idx=k_fixed, latent_value=fixed_value)]
            batch_var = np.array([get_ginni_variance_conti(batch_latent_conti[:, v]) for v in range(latent_conti.shape[1])]) 
            batch_var_norm = np.divide(batch_var, var+eps)
            count[np.argmin(batch_var_norm)][idx]+=1
    if print_option:
        print(count)
    #print(count)
    return get_majority_vote_accuracy(count)

def DisentanglemetricFactorMultiJointMaskcustom(latent_conti, latent_cat_list, mi_array, nclasses, sampler, nvote=800, nex_per_vote=100, eps=1e-10, thr=0.1, print_option=False): 
    mask = np.where(mi_array>thr)[0]
    latent_conti = latent_conti[:, mask]
    if print_option:
        print(mask)

    var = np.array([get_ginni_variance_conti(latent_conti[:, v]) for v in range(latent_conti.shape[1])]+[get_ginni_variance_discrete(array=v) for v in latent_cat_list])
    
    k_set = list()
    for idx in range(nclasses.shape[0]):
        if nclasses[idx]>1: k_set.append(idx)

    if print_option:
        print(k_set)

    nfactor = len(k_set)
    nlatent = var.shape[0]
    nvote_per_factor = int(nvote/nfactor)
    
    count = np.zeros([nlatent, nfactor])
    for idx in range(nfactor):
        for iter_ in range(nvote_per_factor):
            k_fixed = k_set[idx]
            fixed_value = np.random.randint(nclasses[k_fixed])
            sample_idx = sampler(batch_size=nex_per_vote, latent_idx=k_fixed, latent_value=fixed_value)
            batch_latent_conti = latent_conti[sample_idx]
            batch_var = np.array([get_ginni_variance_conti(batch_latent_conti[:, v]) for v in range(latent_conti.shape[1])]+[get_ginni_variance_discrete(array=v[sample_idx]) for v in latent_cat_list])
            batch_var_norm = np.divide(batch_var, var+eps)
            count[np.argmin(batch_var_norm)][idx]+=1
    if print_option:
        print(count)

    return get_majority_vote_accuracy(count)

def get_majority_vote_accuracy(count):
    '''
    Args:
        count - Numpy 2D array [nvoter, ncandidate]
    Return:
        accuracy
    '''
    nvoter = count.shape[0]
    classifier = np.argmax(count, axis=-1)

    nvote = np.sum(count)
    vote_correct = np.sum([count[idx][classifier[idx]] for idx in range(nvoter)]) 
    
    accuracy = vote_correct/nvote
    return accuracy
