from threading import Thread
from queue import Queue

import torch



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
    x: (batch, coorddim)
    v: (batch, coorddim)
    drag: (coorddim,)
    a: (-1, coorddim) -> (-1, coorddim)
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
    def generate_trajectory(self, batch, N, x=None, v=None):
        """ generate trajectory from initial conditions x, v
            WARNING: initial condition tensors *will* be overwritten
            x: (batch, self.dim)
            v: (batch, self.dim)
            x_traj: (batch, N, self.dim)
            v_traj: (batch, N, self.dim) """
        x_traj = torch.zeros((batch, N, self.dim), device="cuda", dtype=torch.float64)
        v_traj = torch.zeros((batch, N, self.dim), device="cuda", dtype=torch.float64)
        if x is None: x = torch.zeros((batch,) + self.drag.shape, dtype=torch.float64, device="cuda")
        if v is None: v = torch.zeros((batch,) + self.drag.shape, dtype=torch.float64, device="cuda")
        for i in range(N):
            vvel_lng_batch(x, v, self.acc_fn, self.drag, self.T, self.dt, self.t_res)
            x_traj[:, i] = x
            v_traj[:, i] = v
        return x_traj, v_traj


def get_polymer_a(k, n, dim=3):
    """ Get an acceleration function defining a polymer system with n atoms and spring constant k
    Shapes:
    x: (batch, n*dim)
    a: (batch, n*dim) """
    def a(x):
        x = x.reshape(-1, n, dim)
        ans = torch.zeros_like(x)
        ans[:, 1:] += k*(x[:, :-1] - x[:, 1:])
        ans[:, :-1] += k*(x[:, 1:] - x[:, :-1])
        return ans.reshape(-1, n*dim)
    return a

def get_polymer_a_cos_potential(k, n, A, p, dim=3):
    """ Get an acceleration function defining a polymer system with n atoms and spring constant k
    All atoms are in a cosine wave potential with amplitude A and wavenumber p
    Shapes:
    x: (batch, n*dim)
    a: (batch, n*dim)
    A: ()
    p: (dim) """
    def a(x):
        x = x.reshape(-1, n, dim)
        ans = (A*p*torch.sin((p*x).sum(-1)))[:, :, None]
        ans[:, 1:] += k*(x[:, :-1] - x[:, 1:])
        ans[:, :-1] += k*(x[:, 1:] - x[:, :-1])
        return ans.reshape(-1, n*dim)
    return a

def get_polymer_a_quart(k, n, dim=3):
    """ Get an acceleration function defining a polymer system with n atoms and a quartic bond potential
        the potential is given by the quartic (k/8)(x**2 - 1)**2
    Shapes:
    x: (batch, n*dim)
    a: (batch, n*dim) """
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


sims = {
    "SHO, Langevin": TrajectorySim(
        (lambda x: -x),
        torch.tensor([1.0], dtype=torch.float64), 1.0,
        1.0, 60
    ),
    "SHO, Ornstein Uhlenbeck": TrajectorySim(
        (lambda x: -x),
        torch.tensor([10.], dtype=torch.float64), 1.0,
        3.0, 60
    ),
    "1D Polymer, Ornstein Uhlenbeck": TrajectorySim(
        get_polymer_a(1.0, 12, dim=1),
        torch.tensor([10.]*12, dtype=torch.float64), 1.0,
        1.0, 20,
        metadata={"poly_len": 12, "space_dim": 1}
    ),
    "1D Polymer, Ornstein Uhlenbeck, long": TrajectorySim(
        get_polymer_a(1.0, 12, dim=1),
        torch.tensor([10.]*12, dtype=torch.float64), 1.0,
        140.0, 20*140, # tune the delta_t to equal a time constant for the slowest mode
        metadata={"poly_len": 12, "space_dim": 1}
    ),
    "1D Polymer, Ornstein Uhlenbeck, medium": TrajectorySim(
        get_polymer_a(1.0, 12, dim=1),
        torch.tensor([10.]*12, dtype=torch.float64), 1.0,
        97.0, 16*97, # tune the delta_t so that slowest mode decays by about 1/2
        metadata={"poly_len": 12, "space_dim": 1}
    ),
    "1D Polymer, Ornstein Uhlenbeck, 10": TrajectorySim(
        get_polymer_a(1.0, 12, dim=1),
        torch.tensor([10.]*12, dtype=torch.float64), 1.0,
        10.0, 16*10,
        metadata={"poly_len": 12, "space_dim": 1}
    ),
    "1D Polymer, Ornstein Uhlenbeck, medium, cosine": TrajectorySim(
        get_polymer_a_cos_potential(1.0, 12, 1.0, torch.tensor([3.0], dtype=torch.float64, device="cuda"), dim=1),
        torch.tensor([10.]*12, dtype=torch.float64), 1.0,
        97.0, 16*97, # tune the delta_t so that slowest mode decays by about 1/2
        metadata={"poly_len": 12, "space_dim": 1}
    ),
    "1D Polymer 12, Ornstein Uhlenbeck, t10": TrajectorySim(
        get_polymer_a(1.0, 12, dim=1),
        torch.tensor([10.]*12, dtype=torch.float64), 1.0,
        10.0, 16*10,
        metadata={"poly_len": 12, "space_dim": 1}
    ),
    "1D Polymer 24, Ornstein Uhlenbeck, t10": TrajectorySim(
        get_polymer_a(1.0, 24, dim=1),
        torch.tensor([10.]*24, dtype=torch.float64), 1.0,
        10.0, 16*10,
        metadata={"poly_len": 24, "space_dim": 1}
    ),
    "1D Polymer 36, Ornstein Uhlenbeck, t10": TrajectorySim(
        get_polymer_a(1.0, 36, dim=1),
        torch.tensor([10.]*36, dtype=torch.float64), 1.0,
        10.0, 16*10,
        metadata={"poly_len": 36, "space_dim": 1}
    ),
    "test_quart": TrajectorySim(
      lambda x: -x*(x**2 - 1),
      torch.tensor([1.0], dtype=torch.float64), 1.0,
      1.0, 16*1,
      metadata={"poly_len": 1, "space_dim": 1}
    ),
}


for l in [12, 24, 36, 48]:
  for t in [3, 10, 30, 100]:
    sims["ou_poly_l%d_t%d" % (l, t)] = TrajectorySim(
        get_polymer_a(1.0, l, dim=1),
        torch.tensor([10.]*l, dtype=torch.float64), 1.0,
        float(t), 16*t,
        metadata={"poly_len": l, "space_dim": 1}
      )
    sims["cos_ou_poly_l%d_t%d" % (l, t)] = TrajectorySim(
        get_polymer_a_cos_potential(1.0, l, 2.0, torch.tensor([2.0], dtype=torch.float64, device="cuda"), dim=1),
        torch.tensor([10.]*l, dtype=torch.float64), 1.0,
        float(t), 16*t,
        metadata={"poly_len": l, "space_dim": 1}
      )
    sims["quart_ou_poly_l%d_t%d" % (l, t)] = TrajectorySim(
        get_polymer_a_quart(4.0, l, dim=1),
        torch.tensor([10.]*l, dtype=torch.float64), 1.0,
        #float(t), 16*t,
        1.0, 32,
        metadata={"poly_len": l, "space_dim": 1}
      )



# DATASET GENERATION
def get_dataset(sim, N, L, t_eql=0, subtract_cm=0, x_only=False, x_init=None, v_init=None):
  """ generate data from a simulation sim.
      generates N trajectories of length L, ans has shape (N, L, 2*system_dim)
      t_eql is the number of delta_t to wait to system to equilibriate, data before this is thrown away
      subtract_cm: 0 means don't subtract center of mass, positive number n means subtract center of mass
      and assume spatial dimension is n
      x_only: only return the x coordinate and not velocity
      initial_state: x_init, v_init: (N, state_dim)
      if these are not None, then te  """
  x_traj, v_traj = sim.generate_trajectory(N, L+t_eql, x=x_init, v=v_init)
  x_traj, v_traj = x_traj[:, t_eql:], v_traj[:, t_eql:]
  if subtract_cm > 0:
    x_tmp = x_traj.reshape(N, L, -1, subtract_cm)
    x_tmp = x_tmp - x_tmp.mean(2, keepdim=True)
    x_traj = x_tmp.reshape(N, L, -1)
  if x_only: return x_traj
  return torch.cat([
      x_traj,
      v_traj
    ], dim=2)


def dataset_gen(sim, N, L, t_eql=0, subtract_cm=0, x_only=False):
  """ generate many datasets in a separate thread
      t_eql, subtract_cm, x_only all do the same thing as they do in get_dataset()
      one should use the send() method for controlling this generator, calling
      send(True) if more data will be required and send(False) otherwise """
  data_queue = Queue(maxsize=32) # we set a maxsize to control the number of items taking up memory on GPU
  control_queue = Queue()
  def thread_main():
    while True: # queue maxsize stops us from going crazy here
      next_dataset = get_dataset(sim, 128*N, L, t_eql=t_eql, subtract_cm=subtract_cm, x_only=x_only).to(torch.float32)
      for i in range(0, 128*N, N):
        if not control_queue.empty():
          command = control_queue.get_nowait()
          if command == "halt":
            return
        data_queue.put(next_dataset[i:i+N])
  t = Thread(target=thread_main)
  t.start()
  while True:
    data = data_queue.get()
    halt = yield data # "keep going" is encoded as None, since python requires the first send() to be passed a None anyway
    if halt is not None:
      control_queue.put("halt")
      yield None
      break


# COMPUTE THE THEORETICAL EIGENVALUE #1 FOR THE 1D PROCESS
# v' = -x - drag*v + noise
# if we average out the noise, and set v' = 0, we get: v = -x/drag
# x' = v = -x/drag
# x = exp(-t/drag)
def get_ou_eigen_1d():
  sim = sims["SHO, Ornstein Uhlenbeck"]
  return torch.exp(-sim.delta_t/sim.drag[0]).item()

# similar, but now v is -kx/drag, where k is associated with a Rouse mode
# we'll let the user of this function worry about the value of k, though
def get_poly_eigen_1d():
  sim = sims["1D Polymer, Ornstein Uhlenbeck"]
  return torch.exp(-sim.delta_t/sim.drag[0]).item()

if __name__ == "__main__":
  print("Linear eigenstate decay per time delta_t for 1D Ornstein Uhlenbeck:", get_ou_eigen_1d())
  print("Linear eigenstate decay per time delta_t for Polymer (assuming reference of k=1):", get_poly_eigen_1d())
  # TODO: delete the following lines:
  dataset = get_dataset(sims["1D Polymer, Ornstein Uhlenbeck, medium, cosine"], 10, 10)





