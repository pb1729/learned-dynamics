import importlib

import sys
sys.path.append("/home/phillip/projects/torchenv/src/koopman")

from config import Config, Condition
from train import training_run


SIMTYPE = "3d_quart_ou_poly"
ARCH = "wgan_3d_taxicab"
RUN_ID = "L2"

L_LIST = [12]#[2, 5, 12, 24, 36, 48]
T_LIST = [100]#[3, 10, 30, 100]

arch_specific = {
  "lr_d": 0.0008, "lr_g": 0.00005, "lpen_wt": 1.0,
  "beta_1": 0.5, "beta_2": 0.99, "weight_decay": 0.01,
  "z_scale": 18., "inst_noise_str_r": 0., "inst_noise_str_g": 0.,
  "adim": 64, "vdim": 32, "agroups": 8, "vgroups": 4,
  "rank": 24,
}

for l in L_LIST:
  for t in T_LIST:
    sim_name = SIMTYPE + "_l%d_t%d" % (l, t)
    training_run("models/%s_%s.%s.pt" % (RUN_ID, sim_name, ARCH),
      Config(sim_name, ARCH,
        cond=Condition.COORDS, x_only=True,
        batch=8, simlen=6, t_eql=4,
        nsteps=8192, save_every=512,
        arch_specific=arch_specific))




