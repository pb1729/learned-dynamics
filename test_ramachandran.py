import mdtraj
import numpy as np
import matplotlib.pyplot as plt

from managan.backmap_to_pdb import model_states_to_mdtrajs
from managan.utils import must_be
from predictor_argparse_util import args_to_predictor_list, add_model_list_arg
from plotting_common import approx_squarish_factorize


def get_angles(traj):
  """ return Ramachandran phis and psis for a ModelState trajectory """
  mdtrajecs = model_states_to_mdtrajs(traj)
  phis = np.stack([
      mdtraj.compute_phi(mdtrajec)[1]
      for mdtrajec in mdtrajecs
    ], axis=1) # (L, batch, npep)
  psis = np.stack([
      mdtraj.compute_psi(mdtrajec)[1]
      for mdtrajec in mdtrajecs
    ], axis=1) # (L, batch, npep)
  return phis, psis

def density_plot(angles):
  for prednm in angles:
    phis, psis = angles[prednm]
    plt.scatter(phis.flatten(), psis.flatten(), label=prednm)
  plt.legend()
  plt.xlim(-np.pi, np.pi)
  plt.ylim(-np.pi, np.pi)
  plt.xlabel("φ [rad]")
  plt.ylabel("ψ [rad]")
  plt.show()

def angle_tcorr(angle):
  """ angle: (L, batch, npep)
      ans: (L - 1, npep) """
  L, batch, npep = angle.shape
  return np.stack([
    np.cos(angle[offset:] - angle[:L - offset]).mean((0, 1))
    for offset in range(L - 1)
  ])

def tcorr_plot(angles):
  for prednm in angles:
    L, batch, npep = angles[prednm][0].shape
    break
  plots_x, plots_y = approx_squarish_factorize(2*npep)
  fig, axs = plt.subplots(plots_y, plots_x, figsize=(plots_x * 3, plots_y * 3), squeeze=False)
  axs_flat = axs.flatten()
  for prednm in angles:
    phis, psis = angles[prednm]
    must_be[L], must_be[batch], must_be[npep] = phis.shape
    must_be[L], must_be[batch], must_be[npep] = psis.shape
    tcorr_phis = angle_tcorr(phis)
    tcorr_psis = angle_tcorr(psis)
    for i in range(npep):
      ax_phi = axs_flat[2*i]
      ax_psi = axs_flat[2*i + 1]
      for ax, tcorr in [(ax_phi, tcorr_phis), (ax_psi, tcorr_psis)]:
        ax.set_xlabel("time [steps]")
        ax.plot(tcorr[:, i], label=prednm)
  for i in range(npep):
    ax_phi = axs_flat[2*i]
    ax_psi = axs_flat[2*i + 1]
    ax_phi.set_ylabel(f"bond {i}: <cos(φ(0) - φ(t))>")
    ax_psi.set_ylabel(f"bond {i}: <cos(ψ(0) - ψ(t))>")
    for ax in [ax_phi, ax_psi]:
      ax.set_xlabel("time [steps]")
      ax.legend()
      ax.set_ylim(0., 1.)
  plt.tight_layout()
  plt.show()


def main(args):
  predictors = args_to_predictor_list(args)
  angles = {}
  for predictor in predictors:
    state = predictor.sample_q(args.batch)
    traj = predictor.predict(args.tmax, state)
    phis, psis = get_angles(traj)
    angles[predictor.name] = (phis, psis)
  if args.plot_type == "density":
    density_plot(angles)
  elif args.plot_type == "tcorr":
    tcorr_plot(angles)



if __name__ == "__main__":
  from argparse import ArgumentParser
  parser = ArgumentParser(prog="test_model")
  add_model_list_arg(parser) # -M and -O
  parser.add_argument("--batch", type=int, default=1)
  parser.add_argument("--tmax", type=int, default=100)
  parser.add_argument("--plot_type", type=str, choices=["density", "tcorr"], default="density", help="Type of plot to generate")
  main(parser.parse_args())
