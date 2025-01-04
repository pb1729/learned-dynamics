import torch

from .utils import must_be
from .sim_utils import RegexDict



def vvel_lng_batch(x, v, a, drag, T, dt, nsteps, device="cuda"):
    """ Langevin dynamics with Velocity-Verlet.
    Batched version: compute multiple trajectories in parallel
    This function MUTATES the position (x) and velocity(v) arrays.
    Compute nsteps updates of the system: We're passed x(t=0) and v(t=0) and
    return x(dt*nsteps), v(dt*nsteps). Leapfrog updates are used for the intermediate steps.
    a(x) is the acceleration as a function of the position.
    drag[] is a vector of drag coefficients to be applied to the system.
    T is the temperature in units of energy.
    dt is the timestep size. nsteps should be >= 1.
    Shapes:
    x: (batch, *shape)                    [L]
    v: (batch, *shape)                    [L/T]
    drag: (*shape)                        [/T]
    a: (-1, *shape) -> (-1, *shape)       [L/TT]
    return = x:(batch, *shape), v:(batch, *shape) """
    sqrt_hlf = 0.5**0.5
    noise_coeffs = torch.sqrt(2*drag*T*dt) # noise coefficients for a dt timestep
    def randn():
        return torch.randn_like(x)
    v += 0.5*(dt*(a(x) - drag*v) + sqrt_hlf*noise_coeffs*randn())
    for i in range(nsteps - 1):
        x += dt*v
        v += dt*(a(x) - drag*v) + noise_coeffs*randn()
    x += dt*v
    v += 0.5*(dt*(a(x) - drag*v) + sqrt_hlf*noise_coeffs*randn())
    return x, v


class TrajectorySim:
    def __init__(self, acc_fn, drag, T, delta_t, t_res, metadata=None):
        """ Object representing a physical system for which we can generate trajectories.
        acc_fn : function defining the system, gives acceleration given position
        drag : vector of drag coefficients, also gives the shape of the position vector
        T : temperature
        delta_t : time spacing at which we take samples
        t_res : time resolution, number of individual simulation steps per delta_t
        metadata: dict of additional useful information about the simulation """
        self.acc_fn = acc_fn
        self.drag = drag.to("cuda")
        self.T = T
        self.delta_t = delta_t
        self.t_res = t_res
        self.dt = delta_t/t_res
        self.shape = drag.shape
        if metadata is not None:
          for key in metadata:
            setattr(self, key, metadata[key])
    def generate_trajectory(self, x, v, time, ret=True):
        """ generate trajectory from initial conditions x, v
            WARNING: initial condition tensors x,v *will* be overwritten
            x: (batch, *self.shape)
            v: (batch, *self.shape)
            ans: (batch, time, *self.shape) """
        batch,          *must_be[self.shape] = x.shape
        must_be[batch], *must_be[self.shape] = v.shape
        if ret:
            ans = torch.zeros((batch, time, *self.shape), device="cuda", dtype=x.dtype)
        for i in range(time):
            vvel_lng_batch(x, v, self.acc_fn, self.drag, self.T, self.dt, self.t_res)
            if ret: ans[:, i] = x
        if ret:
            return ans
    def sample_equilibrium(self, batch, iterations, t_noise=None, t_ballistic=None,
            drag_const=20.): # TODO: come up with a better way to pick the drag constant?
        """ Do our best to sample from the equilibrium distribution by alternately setting drag to be high/zero. """
        if t_noise is None:
            t_noise = self.delta_t
        if t_ballistic is None:
            t_ballistic = self.delta_t
        high_drag = torch.zeros_like(self.drag) + drag_const
        zero_drag = torch.zeros_like(self.drag)
        # start from 0:
        x = torch.zeros((batch,) + self.shape, dtype=self.drag.dtype, device="cuda")
        v = torch.zeros((batch,) + self.shape, dtype=self.drag.dtype, device="cuda")
        # do several iterations:
        for i in range(iterations):
            vvel_lng_batch(x, v, self.acc_fn, high_drag, self.T, self.dt, int(t_noise/self.dt))
            vvel_lng_batch(x, v, self.acc_fn, zero_drag, self.T, self.dt, int(t_ballistic/self.dt))
        # then cooldown with the regular amount of drag for a bit:
        vvel_lng_batch(x, v, self.acc_fn, self.drag, self.T, self.dt, self.t_res)
        return x, v



def get_polymer_a(k, n, dim=3):
    """ Get an acceleration function defining a polymer system with n atoms and spring constant k
    Shapes:
    k: ()             [/TT]
    x: (batch, n, dim) [L]
    a: (batch, n, dim) [L/TT] """
    def a(x):
        ans = torch.zeros_like(x)
        ans[:, 1:] += k*(x[:, :-1] - x[:, 1:])
        ans[:, :-1] += k*(x[:, 1:] - x[:, :-1])
        return ans
    return a

def get_polymer_a_quart(k, n, dim=3):
    """ Get an acceleration function defining a polymer system with n atoms and a quartic bond potential
        the potential is given by the quartic (k/8)(x**2 - 1)**2
    Shapes:
    x: (batch, n, dim) [L]
    a: (batch, n, dim) [L/TT] """
    def bond_force(delta_x):
      return 0.5*k*((delta_x**2).sum(2, keepdim=True) - 1)*delta_x
    def a(x):
        ans = torch.zeros_like(x)
        F = bond_force(x[:, :-1] - x[:, 1:])
        ans[:, 1:]  += F
        ans[:, :-1] -= F
        return ans
    return a

def get_polymer_a_steric(k, n, dim=3, repel_scale=1.0):
    """ Get an acceleration function defining a polymer system, with n atoms and
        a (1/r)**12 repulsive force between all pairs of atoms. """
    def bond_force(delta_x):
      return k*delta_x
    def repel_force(delta_x):
      return -repel_scale*delta_x/((delta_x**2).sum(-1, keepdim=True) + 0.2)**6
    def a(x):
      ans = torch.zeros_like(x)
      F_bond = bond_force(x[:, :-1] - x[:, 1:])
      ans[:, 1:]  += F_bond
      ans[:, :-1] -= F_bond
      ans += repel_force(x[:, :, None] - x[:, None, :]).sum(1)
      return ans
    return a

def get_polymer_a_poten(k, n, dim=3):
    """ Get an acceleration function defining a polymer system with n atoms and spring constant k
    The polymer is in a potential (x**4 + y**4 + z**4)/24
    Shapes:
    k: ()             [/TT]
    x: (batch, n, dim) [L]
    a: (batch, n, dim) [L/TT] """
    def a(x):
        ans = torch.zeros_like(x)
        ans[:, 1:] += k*(x[:, :-1] - x[:, 1:])
        ans[:, :-1] += k*(x[:, 1:] - x[:, :-1])
        ans -= (1/6)*x**3
        return ans
    return a


sims = RegexDict(
  ("ou_sho_t%d", lambda t: TrajectorySim(
    (lambda x: -x),
    torch.tensor([10.], dtype=torch.float32), 1.0,
    float(t), 32*t,
  )),
  ("ou_poly_l%d_t%d", lambda l, t: TrajectorySim(
    get_polymer_a(1.0, l, dim=1),
    10. + torch.zeros(l, 3), 1.0,
    float(t), 16*t,
    metadata={"poly_len": l, "space_dim": 1, "k": 1.0}
  )),
  ("quart_ou_poly_l%d_t%d", lambda l, t: TrajectorySim(
    get_polymer_a_quart(4.0, l, dim=1),
    10. + torch.zeros(l, 1), 1.0,
    float(t), 32*t,
    metadata={"poly_len": l, "space_dim": 1}
  )),
  ("2d_ou_poly_l%d_t%d", lambda l, t: TrajectorySim(
    get_polymer_a(1.0, l, dim=2),
    10. + torch.zeros(l, 2), 1.0,
    float(t), 16*t,
    metadata={"poly_len": l, "space_dim": 2}
  )),
  ("3d_ou_poly_l%d_t%d", lambda l, t: TrajectorySim(
    get_polymer_a(1.0, l, dim=3),
    10. + torch.zeros(l, 3), 1.0,
    float(t), 16*t,
    metadata={"poly_len": l, "space_dim": 3, "k": 1.0}
  )),
  ("3d_quart_ou_poly_l%d_t%d", lambda l, t: TrajectorySim(
    get_polymer_a_quart(4.0, l, dim=3),
    10. + torch.zeros(l, 3), 1.0,
    float(t), 32*t,
    metadata={"poly_len": l, "space_dim": 3}
  )),
  ("3d_ballistic_poly_l%d_t%d", lambda l, t: TrajectorySim(
    get_polymer_a(1.0, l, dim=3),
    torch.zeros(l, 3), 1.0,
    float(t), 16*t,
    metadata={"poly_len": l, "space_dim": 3}
  )),
  ("3d_repel_ou_poly_l%d_t%d", lambda l, t: TrajectorySim(
    get_polymer_a_steric(1.0, l, dim=3),
    10. + torch.zeros(l, 3), 1.0,
    float(t), 32*t,
    metadata={"poly_len": l, "space_dim": 3}
  )),
  ("3d_repel2_ou_poly_l%d_t%d", lambda l, t: TrajectorySim(
    get_polymer_a_steric(1.0, l, dim=3, repel_scale=3.),
    10. + torch.zeros(l, 3), 1.0,
    float(t), 32*t,
    metadata={"poly_len": l, "space_dim": 3}
  )),
  ("3d_repel3_ou_poly_l%d_t%d", lambda l, t: TrajectorySim(
    get_polymer_a_steric(1.0, l, dim=3, repel_scale=10.),
    10. + torch.zeros(l, 3), 1.0,
    float(t), 64*t,
    metadata={"poly_len": l, "space_dim": 3, "k": 1.0}
  )),
  ("3d_repel4_ou_poly_l%d_t%d", lambda l, t: TrajectorySim(
    get_polymer_a_steric(4.0, l, dim=3, repel_scale=10.),
    10. + torch.zeros(l, 3), 1.0,
    float(t), 64*t,
    metadata={"poly_len": l, "space_dim": 3, "k": 4.0}
  )),
  ("3d_poten_ou_poly_l%d_t%d", lambda l, t: TrajectorySim(
    get_polymer_a_poten(1.0, l, dim=3),
    10. + torch.zeros(l, 3), 1.0,
    float(t), 32*t,
    metadata={"poly_len": l, "space_dim": 3, "k": 1.0}
  )),
)


try:
  from polymer_sims_cuda import polymer_sim, SimId
except ModuleNotFoundError:
  print("This package's polymer_sims_cuda extension is not installed, skipping.")
else:
  class CUDASim:
    def __init__(self, sim_id, drag, T, l, delta_t, t_res, metadata=None):
      self.sim_id = sim_id
      self.drag = drag.to("cuda")
      self.T = T
      self.l = l
      self.delta_t = delta_t
      self.t_res = t_res
      self.dt = delta_t/t_res
      self.shape = (l, 3)
      if metadata is not None:
        for key in metadata:
          setattr(self, key, metadata[key])
    def _integrate(self, x, v, drag, T, steps):
      polymer_sim(self.sim_id, x, v, drag, T, self.dt, steps)
    def generate_trajectory(self, x, v, time, ret=True):
      """ generate trajectory from initial conditions x, v
          WARNING: initial condition tensors *will* be overwritten
          we construct and return the trajectory if ret is True.
          x: (batch, self.dim)
          v: (batch, self.dim)
          ans: (batch, time, self.dim) """
      batch,          must_be[self.l], must_be[3] = x.shape
      must_be[batch], must_be[self.l], must_be[3] = v.shape
      if ret:
        ans = torch.zeros((batch, time, *self.shape), device="cuda")
      for i in range(time):
        self._integrate(x, v, self.drag, self.T, self.t_res)
        if ret: ans[:, i] = x
      if ret:
        return ans
    def sample_equilibrium(self, batch, iterations, t_noise=None, t_ballistic=None,
              drag_const=20.): # TODO: come up with a better way to pick the drag constant?
      """ Do our best to sample from the equilibrium distribution by alternately setting drag to be high/zero. """
      if t_noise is None:
        t_noise = self.delta_t
      if t_ballistic is None:
        t_ballistic = self.delta_t
      high_drag = self.drag + drag_const
      zero_drag = torch.zeros_like(self.drag)
      # start from 0:
      x = torch.randn(batch, *self.shape, device="cuda") # spread out the particles initially for numerical stability
      v = torch.zeros(batch, *self.shape, device="cuda")
      # do several iterations:
      for i in range(iterations):
        self._integrate(x, v, high_drag, self.T, int(t_noise/self.dt))
        self._integrate(x, v, zero_drag, self.T, int(t_ballistic/self.dt))
      # then cooldown with the regular amount of drag for a bit:
      self._integrate(x, v, self.drag, self.T, self.t_res)
      return x, v
  sims.add_constructor("repel5_l%d_t%d", lambda l, t: CUDASim(SimId.REPEL5,
    torch.tensor([1.]*l), 1.0, l, float(t), 100*t,
    metadata={"poly_len": l, "space_dim": 3}
  ))
  sims.add_constructor("repel5a_l%d_t%d", lambda l, t: CUDASim(SimId.REPEL5,
    torch.tensor([1.]*l), 1.0, l, float(t), 200*t,
    metadata={"poly_len": l, "space_dim": 3}
  ))





# DATASET GENERATION

def equilibrium_sample(config, batch):
  """ sample from eql. dist of config.sim
      return: tuple(x, v)
      x, v: (batch, dim) """
  return config.sim.sample_equilibrium(batch, config.t_eql)

def get_dataset(config, xv_init, L):
  """ generate data from a simulation. creates a batch of trajectories of length L.
      xv_init: tuple(x_init, v_init)
      x_init: (batch, dim)
      v_init: (batch, dim)
      return x_traj: (batch, L, dim)
      we use the config's "subtract_cm" field to determine the function's behaviour.
      subtract_cm: False means don't subtract center of mass, True means subtract center
      of mass. (requires sim to define a space_dim) """
  x_init, v_init = xv_init
  batch,          must_be[config.sim.dim] = x_init.shape
  must_be[batch], must_be[config.sim.dim] = v_init.shape
  x, v = x_init.clone(), v_init.clone() # prevent ourselves from overwriting inital condition tensors!
  x_traj, v_traj = config.sim.generate_trajectory(x, v, L)
  if config.subtract_mean:
    x_tmp = x_traj.reshape(batch, L, config.sim.poly_len, config.sim.space_dim)
    x_tmp = x_tmp - x_tmp.mean(2, keepdim=True)
    x_traj = x_tmp.reshape(batch, L, -1)
  if config.x_only:
    return x_traj
  else:
    return torch.cat([x_traj, v_traj], dim=2)




# NOTES ON HOW TO COMPUTE THE THEORETICAL TIME CONSTANT FOR AN OU-TYPE PROCESS:
# v' = -k*x - drag*v + noise
# if we average out the noise, and set v' = 0, we get: v = -k*x/drag
# x' = v = -k*x/drag
# thus, the time constant is proportional to drag/k
# (note that k is normalize by particle mass, so its units are [L/TTL]=[/TT])
# interestingly, time constant doesn't depend on temperature!

def get_poly_tc(sim, mode_k):
  """ get the relaxation time constant for a particular mode, with mode spring constant mode_k [/TT].
      WARNING: ASSUMES that the drag is identical for all coordinates! """
  drag = sim.drag[0].item() # assume that the drag for all atoms is identical!
  return drag / mode_k
