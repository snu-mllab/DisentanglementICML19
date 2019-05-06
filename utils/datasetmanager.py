import os
import sys
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..'))

from config.path import DSPRITESPATH
from utils.reader_op import read_npy
from utils.datamanager import DspritesManager

import numpy as np

def dsprites_manager():
    dataset_zip = read_npy(DSPRITESPATH+'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz', encoding='latin1')
    dm = DspritesManager(dataset_zip)
    return dm

