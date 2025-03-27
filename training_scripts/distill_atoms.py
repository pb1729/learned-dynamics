import sys
from os import path
sys.path.append(path.join(path.split(__file__)[0], path.pardir))

from managan.config import Config
from managan.train import training_run


SIMTYPE = "SEQ_t500_L20_seqAAA"#"A_t2000_L40_m20_M45"
ARCH = "wgan_atoms_distill"
RUN_ID = "UU2"

NSTEPS_LIST = [512*i for i in range(1, 33)]


sim_name = SIMTYPE
training_run("models/%s_%s.%s.pt" % (RUN_ID, sim_name, ARCH),
  Config("openmm:" + sim_name, ARCH,
    batch=1, simlen=20,
    nsteps=NSTEPS_LIST, save_every=128,
    arch_specific={
      "lr_d": 0.0002,  "lr_d_fac": 0.98,
      "lr_g": 0.0001, "lr_g_fac": 0.95,
      "lpen_wt": 1.0,
      "beta_1": 0., "beta_2": 0.99, "weight_decay": 0.001,
      "hinge": True, "hinge_leak": 0.1,
      # interval stuff
      "ndiscs": 4,
      # arch params:
      "depth": 4,
      "dim_a_atm": 64, "dim_v_atm": 48, "dim_d_atm": 32,
      "dim_a_amn": 128, "dim_v_amn": 96, "dim_d_amn": 64,
      "groups_a": 8, "groups_v": 8, "groups_d": 8,
      # ACE stuff
      "r_cut_amn": 8.9,
      "r_cut_atm": 4.1,
      "r_cut_bond": 3.9,
      "rank": 32,
      # distillation stuff
      "distill_lambdas": {},
      "distill_models": {
      }
    }))

