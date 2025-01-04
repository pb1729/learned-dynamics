import sys
from os import path
sys.path.append(path.join(path.split(__file__)[0], path.pardir))

from managan.config import Config
from managan.train import training_run


SIMTYPE = "polymer_2"
ARCH = "wgan_3d_newblock4"
RUN_ID = "S4"

L_LIST = [48] # [80]
# rouse time for repel5 and repel5a is 317
T_LIST = [2000]#[round(31.7*i) for i in range(1, 11)]
NSTEPS_LIST = [1024, 2048, 3072, 4096, 5120, 6144, 7168, 8192, 9216, 10240, 11264, 12288, 13312, 14336, 15360, 16384]


arch_specific = {
  "lr_d": 0.0002,  "lr_d_fac": 0.98,
  "lr_g": 0.00004, "lr_g_fac": 0.95,
  "lpen_wt": 1.0,
  "beta_1": 0., "beta_2": 0.99, "weight_decay": 0.001,
  "z_scale": [24., 12., 6., 3.],
  "dim_a": 64, "dim_v": 48, "dim_d": 32,
  "groups_a": 8, "groups_v": 8, "groups_d": 8,
  "rank": 32,
  "hinge": True, "hinge_leak": 0.1,
  # multihead stuff
  "heads": 8, # probably don't need this?
  # interval stuff
  "ndiscs": 2,#4,
  # proxattn stuff:
  "r0_list": [2., 3., 4., 6., 8., 12., 16., 24.],
  "kq_dim": (8, 8),
  # graph embedding stuff
  "r_cut": 3.3
}
for l in L_LIST:
  for t in T_LIST:
    sim_name = SIMTYPE + "_l%d_t%d_L200" % (l, t)
    training_run("models/%s_%s.%s.pt" % (RUN_ID, sim_name, ARCH),
      Config("hoomd:" + sim_name, ARCH,
        batch=1, simlen=16, t_eql=4,
        nsteps=NSTEPS_LIST, save_every=128,
        arch_specific=arch_specific))
