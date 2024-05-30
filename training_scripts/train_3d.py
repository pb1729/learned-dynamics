import importlib

import sys
sys.path.append("/home/phillip/projects/torchenv/src/koopman")

from config import Config, Condition
from train import training_run


SIMTYPE = "3d_ou_poly"
ARCH = "wgan_3d_simple"
RUN_ID = "B"

L_LIST = [12, 24, 36, 48]
T_LIST = [3, 10, 30, 100]

arch_specific = {
  "lr_d": 0.0008, "lr_g": 0.0003,
  "beta_1": 0.5, "beta_2": 0.99,
  "z_scale": 10.0, "inst_noise_str_r": 0.3, "inst_noise_str_g": 0.2,
  "adim": 32, "vdim": 16, "rank": 8,
}

for l in L_LIST:
  for t in T_LIST:
    sim_name = SIMTYPE + "_l%d_t%d" % (l, t)
    training_run("models/%s_%s.%s.pt" % (RUN_ID, sim_name, ARCH),
      Config(sim_name, ARCH,
        cond=Condition.COORDS, x_only=True,
        batch=8, simlen=6, t_eql=4,
        nsteps=2048, save_every=512,
        arch_specific=arch_specific))




