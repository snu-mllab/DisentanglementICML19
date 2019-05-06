import os
import sys
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '../..'))

from model import Model

from config.path import subdirs5resultdir, muldir2mulsubdir

from utils.datasetmanager import dsprites_manager
from utils.format_op import FileIdManager

from local_config import local_dsprites_parser, RESULT_DIR, ID_STRUCTURE

import tensorflow as tf
import numpy as np

if __name__ == '__main__':
    NITER = 300000
    PITER = 20000
    SITER = 10000

    parser = local_dsprites_parser()
    args = parser.parse_args() # parameter required for model

    fim = FileIdManager(ID_STRUCTURE)

    np.random.seed(args.rseed)
    FILE_ID = fim.get_id_from_args(args)
    SAVE_DIR, LOG_DIR, ASSET_DIR = subdirs5resultdir(RESULT_DIR, True)
    SAVE_SUBDIR, ASSET_SUBDIR = muldir2mulsubdir([SAVE_DIR, ASSET_DIR], FILE_ID, True)

    dm = dsprites_manager()
    dm.print_shape()

    model = Model(dm, LOG_DIR+FILE_ID+'.log', args)
    model.set_up_train()
    model.initialize()
    model.train(niter=NITER, siter=SITER, piter=PITER, save_dir=SAVE_SUBDIR, asset_dir=ASSET_SUBDIR)
    model.restore(save_dir=SAVE_SUBDIR)
    train_idx = model.start_iter//PITER
    accuracy = model.evaluate()

