import importlib

import sys
sys.path.append("/home/phillip/projects/torchenv/src/koopman")

from config import Config, Condition
from train import training_run


SIMTYPE = "3d_ou_poly"
RUN_ID = "A"

L_LIST = [12]
T_LIST = [3, 10, 30, 100]

for l in L_LIST:
  for t in T_LIST:
    sim_name = SIMTYPE + "_l%d_t%d" % (l, t)
    training_run("models/%s_%s.dmol_wgan.pt" % (RUN_ID, sim_name),
      Config(sim_name, "dmol_wgan",
        cond=Condition.COORDS, x_only=True,
        batch=4, simlen=8, t_eql=4,
        nsteps=8192, save_every=512,
        arch_specific={
          "lr_d": 0.0008, "lr_g": 0.0003,
          "beta_1": 0.5, "beta_2": 0.99,
          "inst_noise_str_r": 0.3, "inst_noise_str_g": 0.2,
        }))





