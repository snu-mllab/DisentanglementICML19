import argparse

def dsprites_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--gpu", default = 0, help="Utilize which gpu", type = int)
    parser.add_argument("--nbatch", default = 64, help="size of batch", type = int)
    parser.add_argument("--dataset", default = 'dsprites', help="dataset to be used", type = str)

    return parser
