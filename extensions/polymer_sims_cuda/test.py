import time
import torch
import numpy as np

from polymer_sims_cuda import polymer_sim, SimId
from atoms_display import launch_atom_display

# Set up initial conditions
batch = 32
n_particles = 200
x = torch.randn(batch, n_particles, 3, device='cuda')
v = torch.zeros(batch, n_particles, 3, device='cuda')
drag = torch.ones(n_particles, device='cuda') * 1.
T = 1.0
dt = 0.01
nsteps = 100

def clean_for_display(x):
    """ Return first element of batch as a np array on cpu. """
    return x[0].cpu().numpy()

# launch the display
display = launch_atom_display(5*np.ones(n_particles, dtype=int),
    clean_for_display(x), radii_scale=1.)

# run the simulation
while True:
    display.update_pos(clean_for_display(x))
    polymer_sim(SimId.REPEL5, x, v, drag, T, dt, nsteps) # mutates x, v!
    time.sleep(0.1)
