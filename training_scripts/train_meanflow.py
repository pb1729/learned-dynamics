import sys
from os import path
sys.path.append(path.join(path.split(__file__)[0], path.pardir))

from managan.config import Config
from managan.train import training_run


SIMTYPE = "set_seqAAA_10ps_chunklen128"
ARCH = "meanflow_painn7"
RUN_ID = "MF10"
STRIDE = 8

NSTEPS_LIST = [1024*i for i in range(1, 257)]

sim_name = SIMTYPE


training_run("models/%s_%s.%s.pt" % (RUN_ID, sim_name + f"_{STRIDE}", ARCH),
  Config(f"repdataset:{STRIDE}?datasets/" + sim_name, ARCH,
    batch=1, simlen=15,
    nsteps=NSTEPS_LIST,
    save_every=128,
    arch_specific={
      "lr": 0.00003,  "lr_fac": 0.998,
      "beta_1": 0., "beta_2": 0.99, "weight_decay": 0.001,
      # CPaiNN stuff
      "r0": 5., "n": 128,
      "natoms": 15, # Number of atoms in AAA that are not H or OXT
      # diffuser stuff
      "sigma_0": 7.0, # [angstrom]
      "loss_wt": "1",
    }))
