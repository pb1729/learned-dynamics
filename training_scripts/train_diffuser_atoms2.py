import sys
from os import path
sys.path.append(path.join(path.split(__file__)[0], path.pardir))

from managan.config import Config
from managan.train import training_run


SIMTYPE = "set_seqAAA_10ps_chunklen128"#"A_t2000_L40_m20_M45"
ARCH = "diffuser_atoms13"
RUN_ID = "Z68"
STRIDE = 8

NSTEPS_LIST = [1024*i for i in range(1, 65)]


sim_name = SIMTYPE
training_run("models/%s_%s.%s.pt" % (RUN_ID, sim_name, ARCH),
  Config(f"stridataset:{STRIDE}?datasets/{SIMTYPE}", ARCH,
    batch=1, simlen=15,
    nsteps=[256],#NSTEPS_LIST,
    save_every=128,
    arch_specific={
      "lr": 0.0001,  "lr_fac": 0.998,
      "beta_1": 0., "beta_2": 0.99, "weight_decay": 0.001,
      "dim_a": 128, "dim_v": 96, "chan": 256,
      "groups_a": 8, "groups_v": 8,
      # ACE stuff
      "r_cut": 5.5,
      # diffuser stuff
      "depth": 4,
      "t_embed_hdim": 64,
      "noise_sched": "cosine",
      "sigma_max": 16.0,
    }))
