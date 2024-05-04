import importlib

import sys
sys.path.append("/home/phillip/projects/torchenv/src/koopman")
from config import Config, Condition

from train_meanpred import training_run as tr_meanpred
from train_wgan import training_run as tr_wgan
from train_vae import training_run as tr_vae



L_LIST = [12, 24, 36, 48]
T_LIST = [3, 10, 30, 100]

for l in L_LIST:
  for t in T_LIST:
    tr_wgan("models/0_quart_ou_poly_l%d_t%d.wgan_dn_conv.pt" % (l, t),
      Config("quart_ou_poly_l%d_t%d" % (l, t), "wgan_dn_conv",
        cond=Condition.COORDS, x_only=True, subtract_mean=1,
        batch=8, simlen=16, t_eql=4))



