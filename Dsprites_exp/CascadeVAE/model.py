import os
import sys
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '../..'))

from utils.general_class import ModelPlugin
from utils.ortools_op import SolveMaxMatching
from utils.visual_op import matrix_image2big_image
from utils.writer_op import write_pkl, write_gif
from utils.tqdm_op import tqdm_range
from utils.eval_op import DisentanglemetricFactorMask, DisentanglemetricFactorJointMask
from utils.np_op import np_softmax

from tfops.transform_op import apply_tf_op, apply_tf_op_multi_output, apply_tf_op_multi_input
from tfops.train_op import get_train_op_v2
from tfops.lr_op import DECAY_DICT, DECAY_PARAMS_DICT
from tfops.nets import encoder1_64, decoder1_64
from tfops.loss import sigmoid_cross_entropy_without_mean, vae_kl_cost_weight

import tensorflow as tf
import numpy as np

class Model(ModelPlugin):
    def __init__(self, dataset, logfilepath, args):
        super().__init__(dataset, logfilepath, args)
        self.build()

    def build(self):
        self.logger.info("Model building starts")
        tf.reset_default_graph()
        tf.set_random_seed(self.args.rseed)

        self.input1 = tf.placeholder(tf.float32, shape = [self.args.nbatch, self.height, self.width, self.nchannel])
        self.epsilon_input = tf.placeholder(tf.float32, shape=[self.args.nbatch, self.args.nconti])
        self.objective = tf.placeholder(tf.float32, shape = [self.args.nbatch, self.args.ncat])
        self.istrain = tf.placeholder(tf.bool, shape= [])

        self.generate_sess()

        self.mcf = SolveMaxMatching(nworkers=self.args.nbatch, ntasks=self.args.ncat, k=1, pairwise_lamb=self.args.plamb)
        # Encoding
        self.encoder_net = encoder1_64
        self.decoder_net = decoder1_64

        # Continuous rep
        self.mean_total, self.stddev_total = tf.split(self.encoder_net(self.input1, output_dim=2*self.args.nconti, scope='encoder', reuse=False)['output'], num_or_size_splits=2, axis=1)
        self.stddev_total = tf.nn.softplus(self.stddev_total)
        self.z_sample = tf.add(self.mean_total, tf.multiply(self.stddev_total, self.epsilon_input))
        self.dec_output = self.decoder_net(z=tf.concat([self.z_sample, self.objective], axis=-1), output_channel=self.nchannel, scope="decoder", reuse=False)['output']

        # Unary vector
        self.rec_cost_vector = sigmoid_cross_entropy_without_mean(labels=self.input1, logits=self.dec_output)

        # Loss
        self.rec_cost = tf.reduce_mean(self.rec_cost_vector)

        self.loss_dict = dict()
        for idx in range(self.args.nconti+1):
            weight = tf.constant(np.array(idx*[self.args.beta_min] + (self.args.nconti-idx)*[self.args.beta_max]), dtype=tf.float32)
            kl_cost = vae_kl_cost_weight(mean=self.mean_total, stddev=self.stddev_total, weight=weight)
            self.loss_dict[idx] = self.rec_cost+kl_cost+tf.losses.get_regularization_loss()

        # Decode
        self.latent_ph = tf.placeholder(tf.float32, shape = [self.args.nbatch, self.args.nconti+self.args.ncat])
        self.dec_output_ph = tf.nn.sigmoid(self.decoder_net(z=self.latent_ph, output_channel=self.nchannel, scope="decoder", reuse=True)['output'])

        self.logger.info("Model building ends")

    def decode(self, latent_input):
        return apply_tf_op(inputs=latent_input, session=self.sess, input_gate=self.latent_ph, output_gate=self.dec_output_ph, batch_size=self.args.nbatch)

    def set_up_train(self):
        self.logger.info("Model setting up train starts")

        if not hasattr(self, 'start_iter'): self.start_iter = 0
        self.logger.info("Start iter: {}".format(self.start_iter))

        decay_func = DECAY_DICT[self.args.dtype]
        decay_params = DECAY_PARAMS_DICT[self.args.dtype][self.args.nbatch][self.args.dptype].copy() 
        decay_params['initial_step'] = self.start_iter

        self.lr, update_step_op = decay_func(**decay_params)
        self.update_step_op = [update_step_op]

        var_list = [v for v in tf.trainable_variables() if 'encoder' in v.name] + [v for v in tf.trainable_variables() if 'decoder' in v.name] 

        self.train_op_dict = dict()
        with tf.control_dependencies(tf.get_collection("update_ops")):
            for idx in range(self.args.nconti+1):
                self.train_op_dict[idx] = get_train_op_v2(tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.9, beta2=0.999), loss=self.loss_dict[idx], var_list=var_list)

        self.logger.info("Model setting up train ends")

    def run_batch(self, train_idx):
        feed_dict = dict()
        feed_dict[self.input1] = self.dataset.next_batch(batch_size=self.args.nbatch)[0]
        feed_dict[self.istrain] = True
        feed_dict[self.epsilon_input] = np.random.normal(size=[self.args.nbatch, self.args.nconti])

        if train_idx<self.args.ntime:
            feed_dict[self.objective] = np.zeros([self.args.nbatch, self.args.ncat])
        else:
            unary = np.zeros([self.args.nbatch, self.args.ncat])
            for idx in range(self.args.ncat):
                feed_dict[self.objective] = np.tile(np.reshape(np.eye(self.args.ncat)[idx], [1,-1]), [self.args.nbatch, 1])
                unary[:,idx] = self.sess.run(self.rec_cost_vector, feed_dict=feed_dict)
            feed_dict[self.objective] = self.mcf.solve(-unary)[1]

        if train_idx>=self.args.ntime:
            idx = min(train_idx, self.args.nconti)
        else: 
            idx = min(train_idx+1, self.args.nconti)
        self.sess.run(self.train_op_dict[idx], feed_dict=feed_dict)

    def train(self, niter, piter, siter, save_dir=None, asset_dir=None):
        self.logger.info("Model training starts")

        final_iter = self.start_iter+niter
        max_accuracy = -1

        for iter_ in tqdm_range(self.start_iter, final_iter):
            train_idx = (iter_ - self.start_iter)//piter
            self.run_batch(train_idx)

            if (iter_+1)%siter==0 or iter_+1==final_iter:
                include_discrete = False if train_idx < self.args.ntime else True
                accuracy = self.evaluate(include_discrete=include_discrete)

                self.latent_traversal_gif(path=asset_dir+'{}.gif'.format(iter_+1), include_discrete=include_discrete)
                if max_accuracy==-1 or max_accuracy<accuracy:
                    self.save(iter_, save_dir)
                    self.logger.info("Save process")
                    max_accuracy = accuracy
        self.logger.info("Model training ends")

    def evaluate(self, print_option=False, include_discrete=False, eps=1e-8, nsample=1024):
        if include_discrete:
            total_mean, total_std, latent_cat = self.get_latent_total()
            return DisentanglemetricFactorJointMask(mean=total_mean, std=total_std, latent_cat=latent_cat, nclasses=self.dataset.latents_sizes, sampler=self.dataset.next_batch_latent_fix_idx, print_option=print_option, ignore_discrete=False)
        else:
            total_mean, total_std = self.get_mean_std()
            return DisentanglemetricFactorMask(mean=total_mean, std=total_std, nclasses=self.dataset.latents_sizes, sampler=self.dataset.next_batch_latent_fix_idx, print_option=print_option)

    def get_mean_std(self):
        total_mean, total_std = apply_tf_op_multi_output(inputs=self.image, session=self.sess, input_gate=self.input1, output_gate_list=[self.mean_total, self.stddev_total], batch_size=self.args.nbatch, train_gate=self.istrain)
        return total_mean, total_std

    def get_latent_total(self):
        total_mean, total_std = self.get_mean_std()
        unary = np.zeros([self.ndata, self.args.ncat])
        for idx in range(self.args.ncat):
            unary[:,idx] = apply_tf_op_multi_input(inputs_list=[self.image, np.zeros([self.ndata, self.args.nconti]), np.tile(np.reshape(np.eye(self.args.ncat)[idx], [1,-1]), [self.ndata, 1])], session=self.sess, input_gate_list=[self.input1, self.epsilon_input, self.objective], output_gate=self.rec_cost_vector, batch_size=self.args.nbatch, train_gate=self.istrain)
        latent_cat = np_softmax(-unary)
        return total_mean, total_std, latent_cat

    def latent_traversal_gif(self, path, include_discrete=False, nimage=50, nmin=-1.0, nmax=1.0):
        gif = list()
        for i in range(nimage):
            value = nmin + (nmax - nmin)*i/nimage
            latent_conti = value*np.eye(self.args.nconti)
            if include_discrete:
                latent_cat = np.eye(self.args.ncat)
                gif.append(matrix_image2big_image(np.concatenate([np.expand_dims(self.decode(latent_input=np.concatenate([latent_conti, np.tile(np.expand_dims(latent_cat[j], axis=0), [self.args.nconti,1])], axis=1)), axis=0) for j in range(self.args.ncat)], axis=0)))
            else:
                latent_cat = np.zeros([self.args.ncat])
                gif.append(matrix_image2big_image(np.expand_dims(self.decode(latent_input=np.concatenate([latent_conti, np.tile(np.expand_dims(latent_cat, axis=0), [self.args.nconti,1])], axis=1)), axis=0)))
        write_gif(content=gif, path=path)


