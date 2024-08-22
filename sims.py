import torch

from utils import must_be



def vvel_lng_batch(x, v, a, drag, T, dt, nsteps, device="cuda"):
    """ Langevin dynamics with Velocity-Verlet.
    Batched version: compute multiple trajectories in parallel
    This function mutates the position (x) and velocity(v) arrays.
    Compute nsteps updates of the system: We're passed x(t=0) and v(t=0) and
    return x(dt*nsteps), v(dt*nsteps). Leapfrog updates are used for the intermediate steps.
    a(x) is the acceleration as a function of the position.
    drag[] is a vector of drag coefficients to be applied to the system.
    T is the temperature in units of energy.
    dt is the timestep size.
    Shapes:
    x: (batch, coorddim)                  [L]
    v: (batch, coorddim)                  [L/T]
    drag: (coorddim,)                     [/T]
    a: (-1, coorddim) -> (-1, coorddim)   [L/TT]
    return = x:(batch, coorddim), v:(batch, coorddim)"""
    assert nsteps >= 1
    assert x.shape == v.shape and drag.shape == x.shape[1:]
    assert x.dtype == torch.float64 and v.dtype == torch.float64
    sqrt_hlf = 0.5**0.5
    noise_coeffs = torch.sqrt(2*drag*T*dt) # noise coefficients for a dt timestep
    def randn():
        return torch.randn(*x.shape, device=device, dtype=torch.float64)
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
        self.dim = drag.flatten().shape[0]
        if metadata is not None:
          for key in metadata:
            setattr(self, key, metadata[key])
    def generate_trajectory(self, x, v, time):
        """ generate trajectory from initial conditions x, v
            WARNING: initial condition tensors *will* be overwritten
            x: (batch, self.dim)
            v: (batch, self.dim)
            x_traj: (batch, time, self.dim)
            v_traj: (batch, time, self.dim) """
        batch,          must_be[self.dim] = x.shape
        must_be[batch], must_be[self.dim] = v.shape
        x_traj = torch.zeros((batch, time, self.dim), device="cuda", dtype=torch.float64)
        v_traj = torch.zeros((batch, time, self.dim), device="cuda", dtype=torch.float64)
        for i in range(time):
            vvel_lng_batch(x, v, self.acc_fn, self.drag, self.T, self.dt, self.t_res)
            x_traj[:, i] = x
            v_traj[:, i] = v
        return x_traj, v_traj
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
        x = torch.zeros((batch,) + self.drag.shape, dtype=torch.float64, device="cuda")
        v = torch.zeros((batch,) + self.drag.shape, dtype=torch.float64, device="cuda")
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
    x: (batch, n*dim) [L]
    a: (batch, n*dim) [L/TT] """
    def a(x):
        x = x.reshape(-1, n, dim)
        ans = torch.zeros_like(x)
        ans[:, 1:] += k*(x[:, :-1] - x[:, 1:])
        ans[:, :-1] += k*(x[:, 1:] - x[:, :-1])
        return ans.reshape(-1, n*dim)
    return a

def get_polymer_a_quart(k, n, dim=3):
    """ Get an acceleration function defining a polymer system with n atoms and a quartic bond potential
        the potential is given by the quartic (k/8)(x**2 - 1)**2
    Shapes:
    x: (batch, n*dim) [L]
    a: (batch, n*dim) [L/TT] """
    def bond_force(delta_x):
      return 0.5*k*((delta_x**2).sum(2, keepdim=True) - 1)*delta_x
    def a(x):
        x = x.reshape(-1, n, dim)
        ans = torch.zeros_like(x)
        F = bond_force(x[:, :-1] - x[:, 1:])
        ans[:, 1:]  += F
        ans[:, :-1] -= F
        return ans.reshape(-1, n*dim)
    return a

def get_polymer_a_steric(k, n, dim=3, repel_scale=1.0):
    """ Get an acceleration function defining a polymer syste, with n atoms and
        a (1/r)**12 repulsive force between all pairs of atoms. """
    def bond_force(delta_x):
      return k*delta_x
    def repel_force(delta_x):
      return -repel_scale*delta_x/((delta_x**2).sum(-1, keepdim=True) + 0.2)**6
    def a(x):
      x = x.reshape(-1, n, dim)
      ans = torch.zeros_like(x)
      F_bond = bond_force(x[:, :-1] - x[:, 1:])
      ans[:, 1:]  += F_bond
      ans[:, :-1] -= F_bond
      ans += repel_force(x[:, :, None] - x[:, None, :]).sum(1)
      return ans.reshape(-1, n*dim)
    return a

def get_polymer_a_poten(k, n, dim=3):
    """ Get an acceleration function defining a polymer system with n atoms and spring constant k
    The polymer is in a potential (x**4 + y**4 + z**4)/24
    Shapes:
    k: ()             [/TT]
    x: (batch, n*dim) [L]
    a: (batch, n*dim) [L/TT] """
    def a(x):
        x = x.reshape(-1, n, dim)
        ans = torch.zeros_like(x)
        ans[:, 1:] += k*(x[:, :-1] - x[:, 1:])
        ans[:, :-1] += k*(x[:, 1:] - x[:, :-1])
        ans -= (1/6)*x**3
        return ans.reshape(-1, n*dim)
    return a


sims = {}

for t in [3, 10, 30, 100, 300]:
  sims["ou_sho_t%d" % t] = TrajectorySim(
      (lambda x: -x),
      torch.tensor([10.], dtype=torch.float64), 1.0,
      float(t), 32*t,
    )
  for l in [2, 5, 12, 24, 36, 48]:
    sims["ou_poly_l%d_t%d" % (l, t)] = TrajectorySim(
        get_polymer_a(1.0, l, dim=1),
        torch.tensor([10.]*l, dtype=torch.float64), 1.0,
        float(t), 16*t,
        metadata={"poly_len": l, "space_dim": 1, "k": 1.0}
      )
    sims["quart_ou_poly_l%d_t%d" % (l, t)] = TrajectorySim(
        get_polymer_a_quart(4.0, l, dim=1),
        torch.tensor([10.]*l, dtype=torch.float64), 1.0,
        float(t), 32*t,
        metadata={"poly_len": l, "space_dim": 1}
      )
    sims["2d_ou_poly_l%d_t%d" % (l, t)] = TrajectorySim(
        get_polymer_a(1.0, l, dim=2),
        torch.tensor([10.]*l*2, dtype=torch.float64), 1.0,
        float(t), 16*t,
        metadata={"poly_len": l, "space_dim": 2}
      )
    sims["3d_ou_poly_l%d_t%d" % (l, t)] = TrajectorySim(
        get_polymer_a(1.0, l, dim=3),
        torch.tensor([10.]*l*3, dtype=torch.float64), 1.0,
        float(t), 16*t,
        metadata={"poly_len": l, "space_dim": 3, "k": 1.0}
      )
    sims["3d_quart_ou_poly_l%d_t%d" % (l, t)] = TrajectorySim(
        get_polymer_a_quart(4.0, l, dim=3),
        torch.tensor([10.]*l*3, dtype=torch.float64), 1.0,
        float(t), 32*t,
        metadata={"poly_len": l, "space_dim": 3}
      )
    sims["3d_ballistic_poly_l%d_t%d" % (l, t)] = TrajectorySim(
        get_polymer_a(1.0, l, dim=3),
        torch.tensor([0.]*l*3, dtype=torch.float64), 1.0,
        float(t), 16*t,
        metadata={"poly_len": l, "space_dim": 3}
      )
    sims["3d_repel_ou_poly_l%d_t%d" % (l, t)] = TrajectorySim(
        get_polymer_a_steric(1.0, l, dim=3),
        torch.tensor([10.]*l*3, dtype=torch.float64), 1.0,
        float(t), 32*t,
        metadata={"poly_len": l, "space_dim": 3}
      )
    sims["3d_repel2_ou_poly_l%d_t%d" % (l, t)] = TrajectorySim(
        get_polymer_a_steric(1.0, l, dim=3, repel_scale=3.),
        torch.tensor([10.]*l*3, dtype=torch.float64), 1.0,
        float(t), 32*t,
        metadata={"poly_len": l, "space_dim": 3}
      )
    sims["3d_repel3_ou_poly_l%d_t%d" % (l, t)] = TrajectorySim(
        get_polymer_a_steric(1.0, l, dim=3, repel_scale=10.),
        torch.tensor([10.]*l*3, dtype=torch.float64), 1.0,
        float(t), 64*t,
        metadata={"poly_len": l, "space_dim": 3, "k": 1.0}
      )
    sims["3d_repel4_ou_poly_l%d_t%d" % (l, t)] = TrajectorySim(
        get_polymer_a_steric(4.0, l, dim=3, repel_scale=10.),
        torch.tensor([10.]*l*3, dtype=torch.float64), 1.0,
        float(t), 64*t,
        metadata={"poly_len": l, "space_dim": 3, "k": 4.0}
      )
    sims["3d_poten_ou_poly_l%d_t%d" % (l, t)] = TrajectorySim(
        get_polymer_a_poten(1.0, l, dim=3),
        torch.tensor([10.]*l*3, dtype=torch.float64), 1.0,
        float(t), 32*t,
        metadata={"poly_len": l, "space_dim": 3, "k": 1.0}
      )



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




