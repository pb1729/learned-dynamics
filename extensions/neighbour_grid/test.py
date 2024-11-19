import time
import torch
import numpy as np
import matplotlib.pyplot as plt

from atoms_display import launch_atom_display, ATOM_VALENCE_RADII, ATOM_COLORS
from neighbour_grid_cuda import get_neighbours, edges_read, edges_reduce, get_edges

# Set up initial conditions
batch = 20
n_particles = 60#184
box = (30.5, 33.5, 36.5)
tbox = torch.tensor(box, device="cuda")
x = torch.rand(batch, n_particles, 3, device="cuda")*tbox
neighbours, neighbour_counts = get_neighbours(32, 32, 7., *box, x)
maxdeg = neighbour_counts.max().item()
print("greatest degree", maxdeg, "/", neighbours.shape[2])
neighbours = neighbours[:, :, :maxdeg].contiguous()
print(neighbours.shape)
print("average utilization:", neighbour_counts.to(torch.float32).mean().item()/maxdeg)

print("Testing graph symmetry...")
adj = torch.zeros(batch, n_particles, n_particles, dtype=torch.int8)
for b in range(batch):
  for n in range(n_particles):
    for j in range(neighbour_counts[b, n]):
      adj[b, n, neighbours[b, n, j]] = 1
plt.imshow(adj[0])
plt.show()
plt.imshow(adj[0] ^ adj[0].T)
plt.show()
assert (adj == adj.permute(0, 2, 1)).all(), "some graphs in batch are non-symmetric!"
print("done.")

print("Testing edges read...")
indices = torch.arange(n_particles, device="cuda")[None, :, None].expand(batch, -1, -1).contiguous().to(torch.float32)
edge_read_data = edges_read(neighbour_counts, neighbours, indices)
assert abs(neighbours[:, :, :, None] - edge_read_data).sum() < 0.0001, "detected difference!"
print("done.")

print("Testing edges reduce...")
oneses = torch.ones(batch, n_particles, 7, device="cuda")
node_data = edges_read(neighbour_counts, neighbours, oneses)
edge_reduce_data = edges_reduce(neighbour_counts, neighbours, node_data)
assert abs(edge_reduce_data[:, :, 0] - neighbour_counts).sum() < 0.0001, "detected difference!"
print("done.")

print("Testing via display...")
batch_idx = 3
X = x[batch_idx].cpu().numpy()
neighbours = neighbours[batch_idx].cpu().numpy()
neighbour_counts = neighbour_counts[batch_idx].cpu().numpy()
npbox = np.array(box)


atomic_numbers = 6*np.ones(n_particles, dtype=int)

additional_atoms = []
additional_atomic_numbers = []
for i in range(n_particles):
  pos = X[i]
  for j in range(neighbour_counts[i]):
    k = neighbours[i, j]
    pos_other = X[k]
    for t in range(1 + int(i < k), 20, 2):
      tau = t/20
      separation_vec = (pos_other - pos + 0.5*npbox)%npbox - 0.5*npbox
      additional_atoms.append((pos + tau*separation_vec)%npbox)
      additional_atomic_numbers.append(1 + int(i < k))

X = np.concatenate([X, np.array(additional_atoms)])
atomic_numbers = np.concatenate([atomic_numbers, np.array(additional_atomic_numbers)])


# override helium color and radius for display purposes:
ATOM_VALENCE_RADII.flags.writeable = True
ATOM_VALENCE_RADII[2] = 0.23
ATOM_COLORS[2] = np.array([1., 0., 0.])
# launch the display
display = launch_atom_display(atomic_numbers,
    X, radii_scale=1.)


input("Kill existing display before we continue...")
print("Testing get_edges via display")

src, dst, edge_counts = get_edges(32, 32, 7., *box, x)
print(src[0])
print(dst[0])

X = x[batch_idx].cpu().numpy()
SRC = src[(0 if batch_idx == 0 else edge_counts[batch_idx - 1]):edge_counts[batch_idx]].cpu().numpy() - batch_idx*n_particles
DST = dst[(0 if batch_idx == 0 else edge_counts[batch_idx - 1]):edge_counts[batch_idx]].cpu().numpy() - batch_idx*n_particles

atomic_numbers = 6*np.ones(n_particles, dtype=int)
additional_atoms = []
additional_atomic_numbers = []

for a, b in zip(SRC, DST):
  pos_a = X[a]
  pos_b = X[b]
  for t in range(1 + int(a < b), 20, 2):
    tau = t/20
    separation_vec = (pos_b - pos_a + 0.5*npbox)%npbox - 0.5*npbox
    additional_atoms.append((pos_a + tau*separation_vec)%npbox)
    additional_atomic_numbers.append(1 + int(a < b))

for i in range(n_particles):
  pos = X[i]
  for j in range(neighbour_counts[i]):
    k = neighbours[i, j]
    pos_other = X[k]
    for t in range(1 + int(i < k), 20, 2):
      tau = t/20
      separation_vec = (pos_other - pos + 0.5*npbox)%npbox - 0.5*npbox
      additional_atoms.append((pos + tau*separation_vec)%npbox)
      additional_atomic_numbers.append(1 + int(i < k))
X = np.concatenate([X, np.array(additional_atoms)])
atomic_numbers = np.concatenate([atomic_numbers, np.array(additional_atomic_numbers)])
display = launch_atom_display(atomic_numbers, X, radii_scale=1.)
