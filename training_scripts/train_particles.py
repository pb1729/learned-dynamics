import importlib

import sys
sys.path.append("/home/phillip/projects/torchenv/src/koopman")

from config import Config
from train import training_run


SIMTYPE = "particles_1"
ARCH = "wgan_3d_particles"
RUN_ID = "R6"

L = 160
N_LIST = [200]
T_LIST = [2000]
NSTEPS_LIST = [512*i for i in range(1, 33)]


arch_specific = {
  "lr_d": 0.001,  "lr_d_fac": 0.98,
  "lr_g": 0.0001, "lr_g_fac": 0.95,
  "lpen_wt": 1.0,
  "beta_1": 0., "beta_2": 0.99, "weight_decay": 0.001,
  "z_scale": 2.,
  "adim": 64, "vdim": 32, "agroups": 8, "vgroups": 4,
  "rank": 24,
  "gp_coeff": 1.,
  "hinge": True, "hinge_leak": 0.1,
  # multihead stuff
  "covar_pen": False, # seems to actually *create* instability these days
  "heads": 8, # probably don't need this?
  # interval stuff
  "ndiscs": 3,
  # proxattn stuff:
  "r0_list": [2., 3., 4., 6., 8., 12., 16., 24.],
  "kq_dim": (8, 8),
  # particles stuff
  "r_cut": 3.0,
}
for n in N_LIST:
  for t in T_LIST:
    sim_name = SIMTYPE + "_n%d_t%d_L%d" % (n, t, L)
    training_run("models/%s_%s.%s.pt" % (RUN_ID, sim_name, ARCH),
      Config("hoomd:" + sim_name, ARCH,
        batch=1, simlen=16, t_eql=4,
        nsteps=NSTEPS_LIST, save_every=128,
        arch_specific=arch_specific))
