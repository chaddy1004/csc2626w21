import torch
import argparse

if torch.cuda.device_count() > 0:
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
