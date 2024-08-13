import importlib

import sys
sys.path.append("/home/phillip/projects/torchenv/src/koopman")

from config import Config, Condition
from train import training_run


SIMTYPE = "3d_ou_poly"
ARCH = "wgan_3d_gp_classic"
RUN_ID = "TODO"

L_LIST = [12]#[2, 5, 12, 24, 36, 48]
T_LIST = [3]#[3, 10, 30, 100, 300]
NSTEPS_LIST = [1024, 2048, 4096, 8192, 16384, 32768, 65536]


if "gan" in ARCH:
  arch_specific = {
    "lr_d": 0.001,  "lr_d_fac": 0.995,
    "lr_g": 0.0001, "lr_g_fac": 0.99,
    "lpen_wt": 1.0,
    "beta_1": 0., "beta_2": 0.99, "weight_decay": 0.001,
    "z_scale": 4.,
    "inst_noise_str_r": 0., "inst_noise_str_g": 0., # TODO: probably get rid of these at some point (need to make another arch for it...)
    "adim": 64, "vdim": 32, "agroups": 8, "vgroups": 4,
    "rank": 24,
    "gp_coeff": 1.,
    # proxattn stuff:
    "r0_list": [2., 3., 4., 6., 8., 12., 16., 24.],
    "kq_dim": (8, 8),
  }
  for l in L_LIST:
    for t in T_LIST:
      sim_name = SIMTYPE + "_l%d_t%d" % (l, t)
      training_run("models/%s_%s.%s.pt" % (RUN_ID, sim_name, ARCH),
        Config(sim_name, ARCH,
          cond=Condition.COORDS, x_only=True,
          batch=8, simlen=6, t_eql=4,
          nsteps=NSTEPS_LIST, save_every=512,
          arch_specific=arch_specific))
elif "meanpred" in ARCH:
  arch_specific = {
    "lr": 0.0008,
    "beta_1": 0.5, "beta_2": 0.99, "weight_decay": 0.01,
    "adim": 64, "vdim": 32, "agroups": 8, "vgroups": 4,
    "rank": 24,
    "loss_x_scale": 0.6,
  }
  for l in L_LIST:
    for t in T_LIST:
      sim_name = SIMTYPE + "_l%d_t%d" % (l, t)
      training_run("models/%s_%s.%s.pt" % (RUN_ID, sim_name, ARCH),
        Config(sim_name, ARCH,
          cond=Condition.COORDS, x_only=False,
          batch=8, simlen=6, t_eql=4,
          nsteps=65536, save_every=512,
          arch_specific=arch_specific))




