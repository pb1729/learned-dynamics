import importlib

import sys
sys.path.append("/home/phillip/projects/torchenv/src/koopman")

from config import Config, Condition
from train import training_run

L_LIST = [12, 24, 36, 48]
T_LIST = [3, 10, 30, 100]

for l in L_LIST:
  for t in T_LIST:
    training_run("models/5_ou_poly_l%d_t%d.vampnet1.pt" % (l, t),
      Config("ou_poly_l%d_t%d" % (l, t), "vampnet1",
        cond=Condition.COORDS, x_only=True, subtract_mean=1,
        batch=1024, simlen=16, t_eql=4,
        nsteps=2048, save_every=512,
        arch_specific={
          "lr": 5e-5, "wd": 0.05,
          "beta_1": 0.5, "beta_2": 0.99,
          "nf": 96, "outdim": 20,
          "tuning_batches": 32,
        }))




