from SELFRec import SELFRec
from util.conf import ModelConf
import argparse
import random
import os
import numpy as np
import torch

parser = argparse.ArgumentParser(description="Used to replace the configuration file path.")
parser.add_argument('--config', '-c', type=str, default=None, help='Path to the configuration file.')
args = parser.parse_args()

def seed_it(seed):
	random.seed(seed)
	os.environ["PYTHONSEED"] = str(seed)
	np.random.seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = True 
	torch.backends.cudnn.enabled = True
	torch.manual_seed(seed)

if __name__ == '__main__':
    if args.config is not None:
        conf = ModelConf(args.config)
    else:
        conf = ModelConf(f'./conf/yelp.yaml')
    seed_it(conf['seed'])

    rec = SELFRec(conf)
    rec.execute()

