import mdtraj
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from managan.backmap_to_pdb import model_states_to_mdtrajs
from managan.utils import must_be
from plotting.predictor_argparse_util import args_to_predictor_list, add_model_list_arg
from plotting.plotting_common import approx_squarish_factorize


# set big font size
matplotlib.rc("font", size=16)


def get_backbone(state):
  """ state: (..., atoms, 3)
      x_N, x_CA, x_C: (..., aminos, 3) """
  x = state.x_npy
  res_inds = state.metadata.residue_indices
  x_N = x[..., res_inds, :]
  x_CA = x[..., res_inds + 1, :]
  x_C = x[..., res_inds + 2, :]
  return x_N, x_CA, x_C

def get_torsion(x1, x2, x3, x4):
  """ x1, x2, x3, x4: (..., 3) """
  # get displacements
  b1 = x2 - x1
  b2 = x3 - x2
  b3 = x4 - x3
  # Normalize vectors
  b1 /= np.linalg.norm(b1, axis=-1, keepdims=True)
  b2 /= np.linalg.norm(b2, axis=-1, keepdims=True)
  b3 /= np.linalg.norm(b3, axis=-1, keepdims=True)
  # Calculate normal vectors to the planes
  n1 = np.cross(b1, b2)
  n2 = np.cross(b2, b3)
  # Calculate the torsion angle
  x = np.sum(n1 * n2, axis=-1)
  y = np.sum(np.cross(n1, n2) * b2, axis=-1)
  return np.arctan2(y, x)

def get_angles(state):
  N, CA, C = get_backbone(state)
  # Calculate phi angles (C_prev-N-CA-C)
  prev_C = C[:, :, :-1, :]  # C from previous residue
  curr_N = N[:, :, 1:, :]
  curr_CA = CA[:, :, 1:, :]
  curr_C = C[:, :, 1:, :]
  phis = get_torsion(prev_C, curr_N, curr_CA, curr_C)
  # Calculate psi angles (N-CA-C-N_next)
  curr_N = N[:, :, :-1, :]
  curr_CA = CA[:, :, :-1, :]
  curr_C = C[:, :, :-1, :]
  next_N = N[:, :, 1:, :]
  psis = get_torsion(curr_N, curr_CA, curr_C, next_N)
  return phis, psis


def heatmap_plot(angles):
  # phi hist
  for prednm in angles:
    phis, psis = angles[prednm]
    # cut off amino acids on the ends with only one angle:
    phis = phis[..., :-1]
    psis = psis[..., 1:]
    # make a 2d grid for both angles and count how many datapoints are in each cell
    nbins = 32  # Number of bins in each dimension
    H, xedges, yedges = np.histogram2d(
        phis.flatten(), psis.flatten(),
        bins=nbins,
        range=[[-np.pi, np.pi], [-np.pi, np.pi]]
    )
    # normalize and take logs
    total_counts = H.sum()
    H = (0.5*nbins/np.pi)**2 * H / total_counts
    H = np.log(H)
    # Create a figure
    plt.figure(figsize=(8, 6))
    # Plot heatmap
    # Set NaN values in the heatmap to black
    cmap = plt.cm.viridis.copy()
    cmap.set_bad('black')
    plt.imshow(
        H.T,
        extent=[-np.pi, np.pi, -np.pi, np.pi],
        origin='lower',
        aspect='auto',
        cmap=cmap,
        vmax=1.5,
        vmin=np.log((0.5*nbins/np.pi)**2 / total_counts),
    )
    # Add colorbar and labels
    plt.colorbar(label='ln(density)')
    plt.xlabel("φ [rad]")
    plt.ylabel("ψ [rad]")
    plt.show()


def get_data_for_predictor(args, predictor):
  phiss = []
  psiss = []
  for i in range(args.resample):
    state = predictor.sample_q(args.batch)
    traj = predictor.predict(args.tmax, state)
    phis, psis = get_angles(traj)
    phiss.append(phis)
    psiss.append(psis)
  return np.concatenate(phiss, axis=1), np.concatenate(psiss, axis=1) # concat along batch dim, creating a larger effective batch



def main(args):
  predictors = args_to_predictor_list(args)
  angles = {}
  for predictor in predictors:
    print(predictor.name)
    angles[predictor.name] = get_data_for_predictor(args, predictor)
    print("done")
  heatmap_plot(angles)


if __name__ == "__main__":
  from argparse import ArgumentParser
  parser = ArgumentParser(prog="test_model")
  add_model_list_arg(parser) # -M and -O
  parser.add_argument("--resample", type=int, default=1, help="number of times to restart from a new state")
  parser.add_argument("--batch", type=int, default=1, help="batch size")
  parser.add_argument("--tmax", type=int, default=100, help="trajectory length to request")
  main(parser.parse_args())
