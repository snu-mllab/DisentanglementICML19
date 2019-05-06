import os
import sys
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..'))

from utils.general_class import DatamanagerPlugin

import numpy as np
import random

class DspritesManager(DatamanagerPlugin):
    ''' https://github.com/deepmind/dsprites-dataset/blob/master/dsprites_reloading_example.ipynb'''
    def __init__(self, dataset_zip):
        self.image = np.expand_dims(dataset_zip['imgs'], axis=-1).astype(float)
        self.latents_values = dataset_zip['latents_values']
        self.latents_classes = dataset_zip['latents_classes']
        self.metadata = dataset_zip['metadata'][()]
        self.latents_sizes = self.metadata['latents_sizes']
        self.nlatent = len(self.latents_sizes) #6
        self.latents_bases = np.concatenate((self.latents_sizes[::-1].cumprod()[::-1][1:], np.array([1,])))
        super().__init__(ndata=self.image.shape[0])

    def print_shape(self):
        print("Image shape : {}({}, max = {}, min = {})".format(self.image.shape, self.image.dtype, np.amax(self.image), np.amin(self.image)))
        print("Latent size : {}".format(self.latents_sizes))

    def normalize(self, nmin, nmax):
        cmin = np.amin(self.image)
        cmax = np.amax(self.image)
        slope = (nmax-nmin)/(cmax-cmin)

        self.image = slope*(self.image-cmin) + nmin
        self.print_shape()

    def latent2idx(self, latents):
        return np.dot(latents, self.latents_bases).astype(int)

    def next_batch_latent_random(self, batch_size):
        samples = np.zeros([batch_size, self.nlatent])
        for lat_i, lat_size in enumerate(self.latents_sizes):
            samples[:, lat_i] = np.random.randint(lat_size, size=batch_size)
        return samples

    def next_batch_latent_fix(self, batch_size, latent_idx, latent_value): 
        samples = self.next_batch_latent_random(batch_size)
        samples[:, latent_idx] = latent_value
        return self.image[self.latent2idx(samples)]

    def next_batch_latent_fix_idx(self, batch_size, latent_idx, latent_value): 
        samples = self.next_batch_latent_random(batch_size)
        samples[:, latent_idx] = latent_value
        return self.latent2idx(samples)

    def next_batch(self, batch_size):
        subidx = self.sample_idx(batch_size)
        return self.image[subidx], self.latents_classes[subidx]
