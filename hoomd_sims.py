from typing_extensions import Callable, Tuple
import gsd.hoomd
import hoomd
import numpy as np
import time

from sims import SimsDict
from utils import must_be


class HoomdSim:
  def __init__(self,
      get_integrator:Callable[[], hoomd.md.Integrator],
      sample_frame: Callable[[int, Tuple[float, float, float]], gsd.hoomd.Frame],
      nsteps: int, n_particles: int, box: Tuple[float, float, float], kT: float):
    self.get_integrator = get_integrator
    self.sample_frame = sample_frame
    self.n_particles = n_particles
    self.box = box
    self.kT = kT
    self.nsteps = nsteps
  def _settle_simulation(self, simulation:hoomd.Simulation):
    """ MUTATES simulation
        returns True if settling failed and we need to retry, False otherwise"""
    old_dt = simulation.operations.integrator.dt
    simulation.operations.integrator.dt = old_dt / 20 # smaller time step improves success odds
    for i in range(4): # could need several retries before the sim settles
      try:
        simulation.run(1024)
      except RuntimeError as e:
        box = np.array(self.box)
        snapshot = simulation.state.get_snapshot()
        snapshot.particles.position[:] = (snapshot.particles.position + 0.5*box) % box - 0.5*box
        simulation.state.set_snapshot(snapshot)
        simulation.state.thermalize_particle_momenta(filter=hoomd.filter.All(), kT=self.kT)
        continue
      else:
        break
    else:
      return True # failed!
    simulation.operations.integrator.dt = old_dt
    simulation.state.thermalize_particle_momenta(filter=hoomd.filter.All(), kT=self.kT)
    return False
  def sample_q(self):
    frame = self.sample_frame(self.n_particles, self.box)
    # using GPU device. it seems to be slower than CPU, due to small system size, but
    # we get weird memory corruption errors when using CPU, so this avoids such problems
    simulation = hoomd.Simulation(device=hoomd.device.GPU(), seed=1)
    simulation.create_state_from_snapshot(frame)
    simulation.operations.integrator = self.get_integrator()
    simulation.state.thermalize_particle_momenta(filter=hoomd.filter.All(), kT=self.kT)
    if self._settle_simulation(simulation): # MUTATES simulation
      return self.sample_q() # must retry, simulation failed to settle
    return simulation
  def step(self, simulation:hoomd.Simulation):
    """ MUTATES simulation """
    simulation.run(self.nsteps)


def particles_1_frame(n_particles: int, box:Tuple[float, float, float]):
  position = (np.random.rand(n_particles, 3) - 0.5)*np.array(box)
  frame = gsd.hoomd.Frame()
  frame.particles.N = n_particles
  frame.particles.position = position
  frame.particles.typeid = [0]*n_particles
  frame.configuration.box = box + (0, 0, 0)
  frame.particles.types = ["A"]
  return frame

def integrator_particles_1_cons(kT):
  def get_integrator_particles_1() -> hoomd.md.Integrator:
    integrator = hoomd.md.Integrator(dt=0.005)
    cell = hoomd.md.nlist.Cell(buffer=0.4)
    lj = hoomd.md.pair.LJ(nlist=cell)
    lj.params[("A", "A")] = {"epsilon":1, "sigma":1}
    lj.r_cut[("A", "A")] = 2.5
    integrator.forces.append(lj)
    nvt = hoomd.md.methods.ConstantVolume(filter=hoomd.filter.All(), thermostat=hoomd.md.methods.thermostats.Bussi(kT=kT))
    integrator.methods.append(nvt)
    return integrator
  return get_integrator_particles_1


hoomd_sims = SimsDict(
  ("particles_1_n%d_t%d_L%d", lambda n, t, L10: HoomdSim(
    integrator_particles_1_cons(1.5), particles_1_frame,
    t, n, (0.1*L10, 0.1*L10, 0.1*L10), 1.5
  ))
)
