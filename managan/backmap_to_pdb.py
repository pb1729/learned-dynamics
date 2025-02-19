import random
import os

import matplotlib.pyplot as plt
import mdtraj

from .utils import must_be, prod
from .config import get_predictor
from .seq2pdbchain.pdb_util import pdb_line
from .seq2pdbchain.amino_data import Atom, structures, letter_code


def model_state_to_pdb(state):
  assert len(state.size) == 0
  x = state.x_npy
  atom_count = 1
  res_count = 1
  lines = []
  for letter, res_base in zip(state.metadata.seq, state.metadata.residue_indices):
    structure = structures[letter_code[letter]][0].atoms[:-1] # remove last atom as it's N_next
    for i, atom in enumerate(structure):
      atom = Atom(atom.name, x[res_base + i], atom.element)
      lines.append(pdb_line(atom, atom_count, res_count, letter_code[letter]))
      atom_count += 1
    res_count += 1
  return "\n".join(lines)


def pdb_to_mdtraj(pdb:str):
  MDTRAJ_HACK_PATH = "/tmp/disk_file_because_mdtraj_only_likes_those_%d.pdb" % random.randint(0, 100000)
  with open(MDTRAJ_HACK_PATH, "w") as f:
    f.write(pdb)
  ans = mdtraj.load_pdb(MDTRAJ_HACK_PATH)
  os.remove(MDTRAJ_HACK_PATH)
  return ans


def model_state_to_mdtraj(state):
  size = state.size
  state = state.reshape(prod(size))
  mdtrajecs = [
    pdb_to_mdtraj(model_state_to_pdb(state[i]))
    for i in range(prod(size))]
  return mdtraj.join(mdtrajecs)
