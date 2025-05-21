import sys
from os import path
sys.path.append(path.join(path.split(__file__)[0], path.pardir))

from managan.config import Config
from managan.train import training_run


SIMTYPE = "set_seqAAA_10ps_chunklen128"
ARCH = "gan_atoms6"
RUN_ID = "GAN1"
STRIDE=8

NSTEPS_LIST = [1024*i for i in range(1, 65)]


sim_name = SIMTYPE
training_run("models/%s_%s.%s.pt" % (RUN_ID, sim_name, ARCH),
  Config(f"stridataset:{STRIDE}?datasets/{SIMTYPE}", ARCH,
    batch=1, simlen=15,
    nsteps=NSTEPS_LIST, save_every=128,
    arch_specific={
      # training
      "lr_d": 0.0002,  "lr_d_fac": 0.98,
      "lr_g": 0.0001, "lr_g_fac": 0.95,
      "lpen_wt": 1.0,
      "beta_1": 0., "beta_2": 0.99, "weight_decay": 0.001,
      "hinge": True, "hinge_leak": 0.1,
      # generator settings
      "gen_initial_noise": 7.,
      # interval stuff
      "ndiscs": 3,
      # arch params:
      "depth": 4,
      "dim_a": 128, "dim_v": 96, "dim_d": 64, "chan": 256,
      "groups_a": 8, "groups_v": 8, "groups_d": 8,
      # ACE stuff
      "r_cut": 4.9,
    }))
