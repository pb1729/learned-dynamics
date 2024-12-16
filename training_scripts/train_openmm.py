import importlib

import sys
sys.path.append("/home/phillip/projects/torchenv/src/koopman")

from config import Config
from train import training_run


SIMTYPE = "A_t6000_L40_m30_M45"
ARCH = "wgan_3d_newblock9_res"
RUN_ID = "T1"

NSTEPS_LIST = [512*i for i in range(1, 33)]


sim_name = SIMTYPE
training_run("models/%s_%s.%s.pt" % (RUN_ID, sim_name, ARCH),
  Config("openmm:" + sim_name, ARCH,
    batch=1, simlen=16, t_eql=4,
    nsteps=NSTEPS_LIST, save_every=128,
    arch_specific={
      "lr_d": 0.0002,  "lr_d_fac": 0.98,
      "lr_g": 0.0001, "lr_g_fac": 0.95,
      "lpen_wt": 1.0,
      "beta_1": 0., "beta_2": 0.99, "weight_decay": 0.001,
      "z_scale": [1., 0.5, 0.25],
      "dim_a": 64, "dim_v": 48, "dim_d": 32,
      "groups_a": 8, "groups_v": 8, "groups_d": 8,
      "hinge": True, "hinge_leak": 0.1,
      # multihead stuff
      "covar_pen": False, # seems to actually *create* instability these days
      "heads": 8, # probably don't need this?
      # interval stuff
      "ndiscs": 2,#4,
      # ACE stuff
      "r_cut": 8.7,
      "rank": 32,
    }))