import sys
from os import path
sys.path.append(path.join(path.split(__file__)[0], path.pardir))

from managan.config import Config
from managan.train import training_run


SIMTYPE = "set_seqAAA_10ps_chunklen128"#"A_t2000_L40_m20_M45"
ARCH = "unroll_distill_atoms6"
RUN_ID = "DIST1"
STRIDE = 8

NSTEPS_LIST = [1024*i for i in range(1, 65)]


sim_name = SIMTYPE
training_run("models/%s_%s.%s.pt" % (RUN_ID, sim_name, ARCH),
  Config(f"stridataset:{STRIDE}?analysis_datasets/{SIMTYPE}", ARCH,
    batch=1, simlen=15,
    nsteps=NSTEPS_LIST, save_every=128,
    arch_specific={
      "beta_1": 0., "beta_2": 0.99, "weight_decay": 0.001,
      #"lr_disc": 3e-4, "gamma_disc": 1e-4,
      "lr_gen": 1e-4, "gamma_gen": 1.5e-4,
      #"lambda_wass": None, "lambda_l2": 0.4,
      "dim_a": 128, "dim_v": 96, "dim_d": 64, "chan": 256,
      "groups_a": 8, "groups_v": 8, "groups_d": 8,
      #"ndiscs": 4,
      #"hinge": True, "hinge_leak": 0.2,
      # ACE stuff
      "r_cut": 4.9,
      # diffuser stuff
      "depth": 6,
      # point to model to distill from
      "distill_model": "models/Z21_set_seqAAA_10ps_chunklen128_8.diffuser_atoms6.pt",
      "diffusion_steps": 20,
    }))





