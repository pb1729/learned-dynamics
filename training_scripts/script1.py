import importlib

import sys
sys.path.append("/home/phillip/projects/torchenv/src/koopman")
from config import Config, Condition

from train_meanpred import training_run as tr_meanpred
from train_wgan import training_run as tr_wgan
from train_vae import training_run as tr_vae


TRAIN_VAE = False
TRAIN_WGAN = True
TRAIN_MEANPRED = False

L_LIST = [12, 24, 36, 48]
T_LIST = [100]#[3, 10, 30, 100]

for l in L_LIST:
  if TRAIN_VAE:
    tr_vae("models/vae_l%d_t%d.vae.pt" % (l, 3),
      Config("ou_poly_l%d_t%d" % (l, 3), "vae",
        cond=Condition.COORDS, x_only=True, subtract_mean=1,
        batch=8, simlen=16, t_eql=4,
        arch_specific={
          "nz":60
        }))
  for t in T_LIST:
    if TRAIN_WGAN:
      tr_wgan("models/1_wgan_dn_conv_l%d_t%d.wgan_dn_conv.pt" % (l, t),
        Config("ou_poly_l%d_t%d" % (l, t), "wgan_dn_conv",
          cond=Condition.COORDS, x_only=True, subtract_mean=1,
          batch=8, simlen=16, t_eql=4))
    if TRAIN_MEANPRED:
      tr_meanpred("models/test_meanpred_rel3_med_l%d_t%d.meanpred_rel3.pt" % (l, t),
        Config("ou_poly_l%d_t%d" % (l, t), "meanpred_rel3",
          cond=Condition.COORDS, x_only=True, subtract_mean=1,
          batch=8, simlen=16, t_eql=4,
          arch_specific={
            "lr": 0.0008,
            "beta_1": 0.5, "beta_2": 0.99,
            "nf": 96
          }))



