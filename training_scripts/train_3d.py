import importlib

import sys
sys.path.append("/home/phillip/projects/torchenv/src/koopman")

from config import Config
from train import training_run


SIMTYPE = "repel5a"
ARCH = "wgan_3d_flash_clone"
RUN_ID = "Q17"

L_LIST = [48] # [80]
# rouse time for repel5 and repel5a is 317
T_LIST = [32]#[round(31.7*i) for i in range(1, 11)]
NSTEPS_LIST = [1024, 2048, 3072, 4096, 5120, 6144, 7168, 8192, 9216, 10240, 11264, 12288, 13312, 14336, 15360, 16384]


arch_specific = {
  "lr_d": 0.001,  "lr_d_fac": 0.98,
  "lr_g": 0.0001, "lr_g_fac": 0.95,
  "lpen_wt": 1.0,
  "beta_1": 0., "beta_2": 0.99, "weight_decay": 0.001,
  "z_scale": 20.,
  "adim": 64, "vdim": 32, "agroups": 8, "vgroups": 4,
  "rank": 24,
  "gp_coeff": 1.,
  "hinge": True, "hinge_leak": 0.1,
  # multihead stuff
  "covar_pen": False, # seems to actually *create* instability these days
  "heads": 8, # probably don't need this?
  # interval stuff
  "ndiscs": 4,
  # proxattn stuff:
  "r0_list": [2., 3., 4., 6., 8., 12., 16., 24.],
  "kq_dim": (8, 8),
}
for l in L_LIST:
  for t in T_LIST:
    sim_name = SIMTYPE + "_l%d_t%d" % (l, t)
    training_run("models/%s_%s.%s.pt" % (RUN_ID, sim_name, ARCH),
      Config(sim_name, ARCH,
        batch=8, simlen=8, t_eql=4,
        nsteps=NSTEPS_LIST, save_every=128,
        arch_specific=arch_specific))
