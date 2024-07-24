import importlib

import sys
sys.path.append("/home/phillip/projects/torchenv/src/koopman")

from config import Config, Condition
from train import training_run



SIMTYPE = "ou_sho"
ARCH = "wgan_nochain_munet"
RUN_ID = "B45"

T_LIST = [10]#[3, 10, 30, 100]


for t in T_LIST:
  sim_name = SIMTYPE + "_t%d" % t
  training_run("models/%s_%s.%s.pt" % (RUN_ID, sim_name, ARCH),
    Config(sim_name, ARCH,
      cond=Condition.COORDS, x_only=True,
      batch=128, simlen=2, t_eql=20,
      nsteps=9728, save_every=512,
      arch_specific={
        "lr_d": 0.02,
        "lr_g": 0.01,
        "weight_decay": 0.05,
        "ndf": 64, "ngf": 16,
      }))




