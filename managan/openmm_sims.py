import io
import os
import random
from typing_extensions import List, Tuple, Callable
import warnings

import numpy as np
import torch

import openmm
from openmm.vec3 import Vec3
from openmm.app import PDBFile, Modeller, ForceField, Simulation, PME, NoCutoff
from openmm.unit import Quantity, angstrom, kelvin, pico, second

from .seq2pdbchain.amino_data import letter_code, structures
from .seq2pdbchain.seq2pdbchain import pdb_chain
from .sim_utils import RegexDict, OpenMMMetadata
from .amino_seq_markov import generate_seq
from .lettergen import chunked_letters


# constants
ANGSTROM = Quantity(1., unit=angstrom)
PICOSEC  = Quantity(1., unit=pico*second)
LETTERS = [letter for letter in letter_code]
RES_CODE = {letter_code[letter]: letter for letter in letter_code}


class XReporter:
  def __init__(self, sim:Simulation, dt):
    self.dt = dt
    self.x = None
    sim.reporters.append(self)
  def describeNextReport(self, simulation):
    """ API queried by openMM """
    #                x     v      F      E      wrap?
    return (self.dt, True, False, False, False, False)
  def report(self, simulation, state):
    x = state.getPositions(asNumpy=True).value_in_unit(angstrom)
    self.x = x


def init_pdb(seq):
  pdb_str = pdb_chain(seq, showprogress=False)
  with io.StringIO(pdb_str) as f:
    return PDBFile(f)

def pdb_check_box(pdb:PDBFile, boxsz:float):
  pos = np.array(pdb.positions/ANGSTROM)
  diffs = pos.max(0) - pos.min(0)
  return np.all(diffs < boxsz)

def hydrogenate_and_solvate(pdb:PDBFile, ff:ForceField, boxsz:float):
  # http://docs.openmm.org/latest/userguide/application/03_model_building_editing.html
  modeller = Modeller(pdb.topology, pdb.positions)
  modeller.addHydrogens(ff)
  modeller.addSolvent(ff, boxSize=Vec3(boxsz, boxsz, boxsz)*angstrom)
  return modeller

def hydrogenate(pdb:PDBFile, ff:ForceField):
  modeller = Modeller(pdb.topology, pdb.positions)
  modeller.addHydrogens(ff)
  return modeller


def get_get_random_seq(lmin:int, lmax:int):
  def get_random_seq():
    length = random.randint(lmin, lmax)
    return "".join([random.choice(LETTERS) for _ in range(length)])
  return get_random_seq

def get_get_mutated_seq(lmin:int, lmax:int, seqs_envvar_nm:str, cutsz:int=10):
  def mutated_seq_generator():
    seqsfnm = os.environ.get(seqs_envvar_nm)
    if seqsfnm is None:
      raise ValueError(f"Environment variable {seqs_envvar_nm} not set")
    with open(seqsfnm, "r") as f:
      seqs = [line.strip() for line in f.readlines()]
    random.shuffle(seqs)
    for seq in seqs:
      if len(seq) < lmin:
        l_desired = random.randint(lmin, lmax)
        mutation = generate_seq(l_desired - len(seq), init=seq)
        yield seq + mutation
      elif len(seq) > lmax:
        ncuts = len(seq)//cutsz
        cuts = [0, len(seq)] + [random.randint(1, len(seq) - 1) for i in range(ncuts)]
        cuts.sort()
        segments = [seq[cuts[i]:cuts[i + 1]] for i in range(ncuts + 1)]
        long_seq = ""
        for seg in segments[:-1]:
          long_seq += seg
          long_seq += generate_seq(random.randint(1, cutsz), init=long_seq)
        long_seq += segments[-1]
        while len(long_seq) > lmax:
          end_idx = random.randint(lmin, lmax)
          yield long_seq[:end_idx]
          long_seq = long_seq[end_idx:]
        if len(long_seq) >= lmin:
          yield long_seq
      else:
        yield seq
  seqgen = mutated_seq_generator()
  def get_mutated_seq():
    return next(seqgen)
  return get_mutated_seq

def get_get_mutated_seq2(lmin:int, lmax:int, seqs_envvar_nm:str):
  def mutated_seq_generator():
    seqsfnm = os.environ.get(seqs_envvar_nm)
    if seqsfnm is None:
      raise ValueError(f"Environment variable {seqs_envvar_nm} not set")
    for seq in chunked_letters(lmin, lmax, seqsfnm):
      yield seq
  seqgen = mutated_seq_generator()
  def get_mutated_seq():
    return next(seqgen)
  return get_mutated_seq


class OpenMMConfig:
  def __init__(self, dt:int, boxsz:float, temp:float, sample_seq:Callable[[], str], ff:ForceField,
      implicit_water=False):
    self.dt = dt # number of simulation steps, in units of [0.002ps]
    self.boxsz = boxsz
    self.temp = temp
    self.sample_seq = sample_seq
    self.ff = ff
    self.implicit_water = implicit_water
  def _sample_pdb(self, seq):
    for _ in range(10):
      pdb = init_pdb(seq)
      if pdb_check_box(pdb, self.boxsz - 1.):
        return pdb
    else:
      print("max retries exceeded in checking pdb box")
      return False
  def _get_sys_metadata(self, modeller:Modeller) -> OpenMMMetadata:
    seq  = []
    atomic_nums = []
    atom_indices = []
    residue_indices = []
    chain = list(modeller.topology.chains())[0] # main chain should be the protein
    for res in chain.residues():
      seq.append(RES_CODE[res.name])
      residue_indices.append(len(atom_indices))
      nonhydrogens = [atom for atom in res.atoms() if atom.element.atomic_number != 1]
      nonhydrogens = [atom for atom in nonhydrogens if atom.name != "OXT"] # don't include OXT so amino acid sized can be uniform
      atoms_ref = structures[res.name][0].atoms
      for atom, atom_ref in zip(nonhydrogens, atoms_ref):
        assert atom_ref.name.replace("N_next", "OXT") == atom.name, "atom ordering in simulation inconsistent with standard order!"
        atom_indices.append(atom.index)
        atomic_nums.append(atom.element.atomic_number)
    return OpenMMMetadata("".join(seq), np.array(atomic_nums), np.array(atom_indices), np.array(residue_indices))
  def _pdb_to_sim(self, pdb):
    if self.implicit_water:
      modeller = hydrogenate(pdb, self.ff)
    else:
      modeller = hydrogenate_and_solvate(pdb, self.ff, self.boxsz)
    metadata = self._get_sys_metadata(modeller)
    integrator = openmm.LangevinMiddleIntegrator(self.temp*kelvin, 1/PICOSEC, 0.002*PICOSEC)
    system = self.ff.createSystem(modeller.topology, nonbondedMethod=(NoCutoff if self.implicit_water else PME))
    simulation = Simulation(modeller.topology, system, integrator)
    simulation.context.setPositions(modeller.positions)
    return metadata, simulation
  def _settle_sim(self, sim:Simulation):
    sim.minimizeEnergy(tolerance=10)
  def sample_q(self, batch) -> Tuple[OpenMMMetadata, List[Simulation], List[XReporter]]:
    seq = self.sample_seq()
    if batch > 1:
      warnings.warn("""WARNING: Using batch size larger than 1 in openmm sims will affect the distribution of chain lengths and sequences!""")
    sims:List[Simulation] = []
    metadata = None
    for i in range(batch):
      pdb = self._sample_pdb(seq)
      if pdb is False:
        # have to give up and retry with a different sequence
        return self.sample_q(batch)
      new_metadata, sim = self._pdb_to_sim(pdb)
      if metadata is not None:
        assert (new_metadata.seq == metadata.seq
          and np.all(new_metadata.atomic_nums == metadata.atomic_nums)
          and np.all(new_metadata.atom_indices == metadata.atom_indices)
          and np.all(new_metadata.residue_indices == metadata.residue_indices))
      metadata = new_metadata
      sims.append(sim)
    for sim in sims:
      self._settle_sim(sim)
    reporters = [XReporter(sim, self.dt) for sim in sims]
    for sim in sims: # load first x positions
      sim.step(1 + self.dt)
    assert metadata is not None, "batch size 0 is invalid"
    return metadata, sims, reporters


# force fields
FF_DEFAULT = ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')
FF_DRY     = ForceField('amber14-all.xml', 'implicit/gbn2.xml')

# define sims
openmm_sims = RegexDict(
  # A_ sims have T at 300K. example: A_t6000_L40_m20_M60
  ("A_t%d_L%d_m%d_M%d", lambda t, L, m, M: OpenMMConfig(t, float(L), 300., get_get_random_seq(m, M), FF_DEFAULT)),
  ("B_t%d_L%d_m%d_M%d", lambda t, L, m, M: OpenMMConfig(t, float(L), 300., get_get_mutated_seq(m, M, "MANAGAN_SEQS_LOC"), FF_DEFAULT)),
  ("C_t%d_L%d_m%d_M%d", lambda t, L, m, M: OpenMMConfig(t, float(L), 300., get_get_mutated_seq2(m, M, "MANAGAN_SEQS_LOC"), FF_DEFAULT)),
  ("SEQ_t%d_L%d_seq%Q", lambda t, L, seq: OpenMMConfig(t, float(L), 300., lambda: seq, FF_DEFAULT)),
  ("SEQ_DRY_t%d_seq%Q", lambda t, seq: OpenMMConfig(t, float("inf"), 300., lambda: seq, FF_DRY, implicit_water=True))
)


