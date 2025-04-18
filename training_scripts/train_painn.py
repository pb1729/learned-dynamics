import sys
from os import path
sys.path.append(path.join(path.split(__file__)[0], path.pardir))

from managan.config import Config
from managan.train import training_run


SIMTYPE = "SEQ_t500_L20_seqAAA"#"A_t2000_L40_m20_M45"
ARCH = "diffuser_painn2"
RUN_ID = "W15"

NSTEPS_LIST = [1024*i for i in range(1, 65)]

sim_name = SIMTYPE
training_run("models/%s_%s.%s.pt" % (RUN_ID, sim_name, ARCH),
  Config("openmm:" + sim_name, ARCH,
    batch=1, simlen=12,
    nsteps=NSTEPS_LIST, save_every=128,
    arch_specific={
      "lr": 0.0001,  "lr_fac": 0.998,
      "beta_1": 0.5, "beta_2": 0.99, "weight_decay": 0.001,
      # CPaiNN stuff
      "r0": 5., "n": 128,
      "natoms": 15, # Number of atoms in AAA that are not H or OXT
      # diffuser stuff
      "sigma_max": 6.0,
    }))
