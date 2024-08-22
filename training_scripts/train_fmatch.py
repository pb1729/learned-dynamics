import importlib

import sys
sys.path.append("/home/phillip/projects/torchenv/src/koopman")

from config import Config, Condition
from train import training_run

SIMTYPE = "ou_sho"
ARCH = "fmatch_line"
RUN_ID = "P10"

T_LIST = [3]
NSTEPS_LIST = [1024, 2048, 4096, 8192]


for t in T_LIST:
  sim_name = SIMTYPE + "_t%d" % t
  training_run("models/%s_%s.%s.pt" % (RUN_ID, sim_name, ARCH),
    Config(sim_name, ARCH,
      cond=Condition.COORDS, x_only=True,
      batch=64, simlen=8, t_eql=4,
      nsteps=NSTEPS_LIST, save_every=512,
      arch_specific={
        "lr": 5e-5, "wd": 0.05,
        "beta_1": 0.5, "beta_2": 0.99,
        "nf": 128, "ngf": 128, "outdim": 10,
        "by_matrix": False,
      }))




