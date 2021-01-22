import time
import sys
import os
import hashlib
import shutil
import numpy as np

# hack argv
sys.argv = sys.argv[2:]
from common import common
args = common.args

from common import ga


str_md5 = hashlib.md5(args.custom_alignment.encode()).hexdigest()

folder = f"output_data/{args.tensorboard}/model-{args.train_bodies_str}-CustomAlignWrapper-md{str_md5}-sd{args.seed}/PPO_1"

if os.path.exists(folder):
    shutil.rmtree(folder)
    print("old results removed.")

# To avoid accidentally create a infinite loop submitting many jobs
time.sleep(1)
exit(0)
