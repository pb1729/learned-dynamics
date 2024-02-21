import numpy as np
import torch


def vvel_lng_batch(x, v, a, drag, T, dt, nsteps):
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
    noise_coeffs = np.sqrt(2*drag*T*dt) # noise coefficients for a dt timestep
    def randn():
        return np.random.randn(*x.shape)
    v += 0.5*(dt*(a(x) - drag*v) + np.sqrt(0.5)*noise_coeffs*randn())
    for i in range(nsteps - 1):
        x += dt*v
        v += dt*(a(x) - drag*v) + noise_coeffs*randn()
    x += dt*v
    v += 0.5*(dt*(a(x) - drag*v) + np.sqrt(0.5)*noise_coeffs*randn())
    return x, v


class TrajectorySim:
    def __init__(self, acc_fn, drag, T, delta_t, t_res):
        """ Object representing a physical system for which we can generate trajectories.
        acc_fn : function defining the system, gives acceleration given position
        drag : vector of drag coefficients, also gives the shape of the position vector
        T : temperature
        delta_t : time spacing at which we take samples
        t_res : time resolution, number of individual simulation steps per delta_t """
        self.acc_fn = acc_fn
        self.drag = drag
        self.T = T
        self.delta_t = delta_t
        self.t_res = t_res
        self.dt = delta_t/t_res
        self.dim = drag.flatten().shape[0]
    def generate_trajectory(self, batch, N, initial_x=None, initial_v=None):
        x_traj = np.zeros((batch, N, self.dim))
        v_traj = np.zeros((batch, N, self.dim))
        if initial_x is None: initial_x = np.zeros((batch,) + self.drag.shape)
        if initial_v is None: initial_v = np.zeros((batch,) + self.drag.shape)
        x = initial_x.copy()
        v = initial_v.copy()
        for i in range(N):
            vvel_lng_batch(x, v, self.acc_fn, self.drag, self.T, self.dt, self.t_res)
            x_traj[:, i] = x
            v_traj[:, i] = v
        return x_traj, v_traj




# POLYMER (LINEAR CHAIN)
polymer_length = 12

def get_polymer_a(k, n, dim=3):
    """ Get an acceleration function defining a polymer system with n atoms and spring constant k
    Shapes:
    x: (batch, n*dim)
    a: (batch, n*dim) """
    def a(x):
        x = x.reshape(-1, n, dim)
        ans = np.zeros_like(x)
        ans[:, 1:] += k*(x[:, :-1] - x[:, 1:])
        ans[:, :-1] += k*(x[:, 1:] - x[:, :-1])
        return ans.reshape(-1, n*dim)
    return a


sims = {
    "1D Ornstein Uhlenbeck": TrajectorySim(
        (lambda x: -x),
        np.array([10.]), 1.0,
        3.0, 60
    ),
    "1D Polymer, Ornstein Uhlenbeck": TrajectorySim(
        get_polymer_a(1.0, polymer_length, dim=1),
        np.array([10.]*polymer_length), 1.0,
        1.0, 20
    ),
}


# DATASET GENERATION
def get_dataset(sim, N, L, device="cuda"):
    dataset = np.zeros((N, L, sim.dim)) # just training on position coordinates for now. training on velocity too makes dim twice as large
    t_eql = 120 # number of delta_t to wait to system to equilibriate, data before this point is thrown away
    x_traj, _ = sim.generate_trajectory(N, L+t_eql)
    dataset = x_traj[:, t_eql:]
    return torch.tensor(dataset, dtype=torch.float32, device=device)

def subtract_cm_1d(dataset):
    """ given a polymer dataset, subtract the mean position (center of mass) of the polymer
        for each position in the dataset. Polymer should be 1d. Inputs:
        dataset: (N, L, polymer_length) """
    return dataset - dataset.mean(2)[:, :, None]

# COMPUTE THE THEORETICAL EIGENVALUE #1 FOR THE 1D PROCESS
# v' = -x - drag*v + noise
# if we average out the noise, and set v' = 0, we get: v = -x/drag
# x' = v = -x/drag
# x = exp(-t/drag)
def get_ou_eigen_1d():
  sim = sims["1D Ornstein Uhlenbeck"]
  return np.exp(-sim.delta_t/sim.drag[0])

# similar, but now v is -kx/drag, where k is associated with a Rouse mode
# we'll let the user of this function worry about the value of k, though
def get_poly_eigen_1d():
  sim = sims["1D Polymer, Ornstein Uhlenbeck"]
  return np.exp(-sim.delta_t/sim.drag[0])

if __name__ == "__main__":
  print("Linear eigenstate decay per time delta_t for 1D Ornstein Uhlenbeck:", get_ou_eigen_1d())
  print("Linear eigenstate decay per time delta_t for Polymer (assuming reference of k=1):", get_poly_eigen_1d())

