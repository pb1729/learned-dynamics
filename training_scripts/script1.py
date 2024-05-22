import importlib

import sys
sys.path.append("/home/phillip/projects/torchenv/src/koopman")

from config import Config, Condition
from train import training_run


TRAIN_WGAN = True
TRAIN_MEANPRED = True
SIMTYPE = "quart_ou_poly"
RUN_ID = "2"

L_LIST = [12, 24, 36, 48]
T_LIST = [3, 10, 30, 100]

for l in L_LIST:
  for t in T_LIST:
    sim_name = SIMTYPE + "_l%d_t%d" % (l, t)
    if TRAIN_MEANPRED:
      training_run("models/%s_%s.meanpred_rel3.pt" % (RUN_ID, sim_name),
        Config(sim_name, "meanpred_rel3",
          cond=Condition.COORDS, x_only=True, subtract_mean=1,
          batch=8, simlen=16, t_eql=4,
          nsteps=8192, save_every=512,
          arch_specific={
            "lr": 0.0008,
            "beta_1": 0.5, "beta_2": 0.99,
            "nf": 96
          }))
    if TRAIN_WGAN:
      training_run("models/%s_%s.wgan_dn_conv.pt" % (RUN_ID, sim_name),
        Config(sim_name, "wgan_dn_conv",
          cond=Condition.COORDS, x_only=True, subtract_mean=1,
          batch=8, simlen=16, t_eql=4,
          nsteps=8192, save_every=512,
          arch_specific={
            "lr_d": 0.0008, "lr_g": 0.0003,
            "beta_1": 0.5, "beta_2": 0.99,
            "ndf": 64, "ngf":48,
            "z_scale": 20, "inst_noise_str_r": 0.3, "inst_noise_str_g": 0.2,
          }))




