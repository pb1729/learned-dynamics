import importlib

import sys
sys.path.append("/home/phillip/projects/torchenv/src/koopman")

from config import Config, Condition
from train import training_run

SIMTYPE = "3d_ou_poly"
ARCH = "fmatch_1"
RUN_ID = "O6"

L_LIST = [12]#[12, 24, 36, 48]
T_LIST = [3]#[3, 10, 30, 100]
NSTEPS_LIST = [1024, 2048, 4096, 8192]

for l in L_LIST:
  for t in T_LIST:
    sim_name = SIMTYPE + "_l%d_t%d" % (l, t)
    training_run("models/%s_%s.%s.pt" % (RUN_ID, sim_name, ARCH),
      Config(sim_name, ARCH,
        cond=Condition.COORDS, x_only=True,
        batch=64, simlen=16, t_eql=4,
        nsteps=NSTEPS_LIST, save_every=512,
        arch_specific={
          "lr": 5e-5, "wd": 0.05,
          "beta_1": 0.5, "beta_2": 0.99,
          "adim": 96, "vdim": 64,
          "agroups": 8, "vgroups": 4,
          "anf": 36, "vnf": 24,
          "rank": 24, "z_scale": 4.,
          # proxattn stuff:
          "r0_list": [2., 3., 4., 6., 8., 12., 16., 24.],
          "kq_dim": (8, 8),
        }))




