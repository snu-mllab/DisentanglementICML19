import os
import sys
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '../..'))

import numpy as np

def params2id(*args):
    nargs = len(args)
    id_ = '{}'+'_{}'*(nargs-1)
    return id_.format(*args)

def print_numpy(array):
    print(','.join([str(array[v]) for v in range(len(array))]))

class FileIdManager:
    def __init__(self, *attrs):
        self.attrs = attrs[0]
        self.nattr = len(self.attrs)

    def get_id_from_args(self, args):
        tmp = list()
        for attr in self.attrs:
            if attr == '*': tmp.append('*')
            elif type(attr)!=str: tmp.append(attr)
            else: tmp.append(getattr(args, attr))
        return params2id(*tuple(tmp))

    def update_args_with_id(self, args, id_):
        id_split = id_.split('_')
        assert len(id_split)==self.nattr, "id_ should be composed of the same number of attributes"

        for idx in range(self.nattr):
            attr = self.attrs[idx]
            type_attr = type(getattr(args, attr))
            setattr(args, attr, type_attr(id_split[idx]))

