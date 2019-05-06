import os
import sys
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '../..'))

from config.path import ROOT
from config.parser import dsprites_parser

KEY = 'CascadeVAE-C'
RESULT_DIR = ROOT+'{}/'.format(KEY)
TEST_ASSET_DIR = './asset/'
ID_STRUCTURE_DICT = {
        'CascadeVAE-C' : ('nbatch', 'nconti', 'beta_min', 'beta_max', 'dptype', 'rseed'),
        }
ID_STRUCTURE = ID_STRUCTURE_DICT[KEY]

def local_dsprites_parser():
    parser = dsprites_parser()
    parser.add_argument("--rseed", default = 0, help="random seed", type = int)
    parser.add_argument("--beta_min", default = 0.1, help="min value for +beta*kl_cost", type = float)
    parser.add_argument("--beta_max", default = 10.0, help="max value for +beta*kl_cost", type = float)
    parser.add_argument("--dtype", default = 'stair', help="decay type", type = str)
    parser.add_argument("--dptype", default = 'a3', help="decay parameter type", type = str)
    parser.add_argument("--nconti", default = 10, help="the dimension of continuous representation", type = int)
    return parser

