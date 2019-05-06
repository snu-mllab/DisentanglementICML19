import pickle
import numpy as np
import os
import imageio
from matplotlib.backends.backend_pdf import PdfPages

def create_dir(dirname):
    if not os.path.exists(dirname):
        print("Creating %s"%dirname)
        os.makedirs(dirname)
    else:
        print("Already %s exists"%dirname)

def create_muldir(*args):
    for dirname in args: create_dir(dirname) 

def write_pkl(content, path):
    with open(path, 'wb') as f:
        print("Pickle is written on %s"%path)
        try: pickle.dump(content, f)
        except OverflowError: pickle.dump(content, f, protocol=4)

def write_npy(content, path):
    print("Numpy is written on %s"%path)
    np.save(path, content)

class MatplotlibPdfManager:
    def __init__(self, path, plt, pad_inches=None):
        self.path = path
        print("Creating {}".format(self.path))
        self.pdf = PdfPages(self.path)
        self.plt = plt
        self.ncount = 0
        self.pad_inches=pad_inches

    def generate_from(self):
        self.ncount+=1
        print("From here generating {} pages in {}".format(self.ncount, self.path))
        self.plt.close()

    def generate_to(self):
        print("To here generating {} pages in {}".format(self.ncount, self.path))
        if self.pad_inches is None:
            self.pdf.savefig(bbox_inches="tight")
        else:
            self.pdf.savefig(pad_inches = self.pad_inches)

def write_gif(content, path):
    print("Create gif on {}".format(path))
    imageio.mimsave(path, content)

