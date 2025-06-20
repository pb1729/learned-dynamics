import sys
from os import path
sys.path.append(path.join(path.split(__file__)[0], path.pardir))

from managan.config import Config
from managan.train import training_run


SIMTYPE = "set_seqAAA_10ps_chunklen128"#"A_t2000_L40_m20_M45"
ARCH = "advae_atoms7"
RUN_ID = "ADVAE2"
STRIDE = 8

NSTEPS_LIST = [1024*i for i in range(1, 65)]


sim_name = SIMTYPE
training_run("models/%s_%s.%s.pt" % (RUN_ID, sim_name, ARCH),
  Config(f"stridataset:{STRIDE}?datasets/{SIMTYPE}", ARCH,
    batch=1, simlen=15,
    nsteps=NSTEPS_LIST, save_every=128,
    arch_specific={
      "lr_ae": 3e-5, "gamma_ae": 3e-5,
      "lr_dn": 3e-4, "gamma_dn": 1e-5,
      "beta_1": 0., "beta_2": 0.99, "weight_decay": 0.001,
      "dim_a": 128, "dim_v": 128, "dim_d": 64, "chan": 256,
      "groups_a": 8, "groups_v": 8, "groups_d": 8,
      # autoencoder stuff
      "zdim_a": 12, "zdim_v": 12,
      "lambda_ae": 5.0,
      # ACE stuff
      "r_cut": 4.9,
      # diffuser stuff
      "depth": 4,
      "t_embed_hdim": 64,
      "eps_lwt": 0.2,
    }))
