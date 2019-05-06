import os
import sys
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '../..'))

from utils.logger_op import LoggerManager
from utils.gpu_op import selectGpuById
from tfops.init_op import rest_initializer

import tensorflow as tf
import numpy as np
import glob

class DatamanagerPlugin:
    def __init__(self, ndata):
        self.ndata = ndata
        self.start, self.end = 0, 0
        self.fullidx = np.arange(self.ndata)

    def sample_idx(self, batch_size):
        if self.start == 0 and self.end ==0:
            np.random.shuffle(self.fullidx) # shuffle first
        
        if self.end + batch_size > self.ndata:
            self.start = self.end
            self.end = (self.end + batch_size)%self.ndata
            subidx = np.append(self.fullidx[self.start:self.ndata], self.fullidx[0:self.end])
            self.start = 0
            self.end = 0
        else:
            self.start = self.end
            self.end += batch_size
            subidx = self.fullidx[self.start:self.end]
        return subidx

class ModelPlugin: 
    def __init__(self, dataset, logfilepath, args):
        self.args = args

        selectGpuById(self.args.gpu)
        self.logfilepath = logfilepath
        self.logger = LoggerManager(self.logfilepath, __name__)
        self.set_dataset(dataset) 

    def set_dataset(self, dataset):
        self.logger.info("Setting dataset starts")
        self.dataset = dataset
        self.image = self.dataset.image 
        self.ndata, self.height, self.width, self.nchannel = self.image.shape
        self.logger.info("Setting dataset ends")

    def build(self, *args, **kwargs):
        """Builds the neural networks"""
        raise NotImplementedError('`build` is not implemented for model class {}'.format(self.__class__.__name__))

    def set_up_train(self, *args, **kwargs):
        """Builds the neural networks"""
        raise NotImplementedError('`set_up_train` is not implemented for model class {}'.format(self.__class__.__name__))

    def train(self, *args, **kwargs):
        """train the neural networks"""
        raise NotImplementedError('`train` is not implemented for model class {}'.format(self.__class__.__name__))

    def generate_sess(self):
        try: self.sess
        except AttributeError:
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess=tf.Session(config=config)

    def initialize(self):
        '''Initialize uninitialized variables'''
        self.logger.info("Model initialization starts")
        rest_initializer(self.sess) 
        self.start_iter = 0
        self.logger.info("Model initialization ends")

    def save(self, global_step, save_dir, reset_option=True):
        self.logger.info("Model save starts")
        if reset_option: 
            for f in glob.glob(save_dir+'*'): os.remove(f)
        saver=tf.train.Saver(max_to_keep = 5)
        saver.save(self.sess, os.path.join(save_dir, 'model'), global_step = global_step)
        self.logger.info("Model save in %s"%save_dir)
        self.logger.info("Model save ends")

    def restore(self, save_dir, restore_iter=-1):
        """Restore all variables in graph with the latest version"""
        self.logger.info("Restoring model starts...")
        saver = tf.train.Saver()
        checkpoint = tf.train.latest_checkpoint(save_dir)

        if restore_iter==-1:
            self.start_iter = int(os.path.basename(checkpoint)[len('model')+1:])
        else:
            self.start_iter = restore_iter
            checkpoint = save_dir+'model-%d'%restore_iter
        self.logger.info("Restoring from {}".format(checkpoint))
        self.generate_sess()
        saver.restore(self.sess, checkpoint)
        self.logger.info("Restoring model done.")        

    def regen_session(self):
        tf.reset_default_graph()
        self.sess.close()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess=tf.Session(config=config)
        
    def delete(self):
        tf.reset_default_graph()
        self.logger.remove()
        del self.logger


