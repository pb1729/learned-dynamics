import mdtraj
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from managan.backmap_to_pdb import model_states_to_mdtrajs
from managan.utils import must_be
from predictor_argparse_util import args_to_predictor_list, add_model_list_arg
from plotting_common import approx_squarish_factorize


PSI_CRIT = 1.5
PHI_CRIT = 0.0#-2.05


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


def density_plot(angles):
  # psi hist
  for prednm in angles:
    phis, psis = angles[prednm]
    # cut off amino acids on the ends with only one angle:
    psis = psis[..., 1:]
    plt.hist(psis.flatten(), bins=32, range=(-np.pi, np.pi), label=prednm[:20], alpha=0.4)
    #print(f"Median ψ for {prednm}: {(np.median((psis.flatten() + np.pi - PSI_CRIT) % (2*np.pi)) + np.pi + PSI_CRIT) % (2*np.pi)}")
  plt.xlabel("ψ [rad]")
  plt.ylabel("counts")
  plt.legend()
  plt.show()
  # phi hist
  for prednm in angles:
    phis, psis = angles[prednm]
    # cut off amino acids on the ends with only one angle:
    phis = phis[..., :-1]
    plt.hist(phis.flatten(), bins=32, range=(-np.pi, np.pi), label=prednm[:20], alpha=0.4)
    #print(f"Median φ for {prednm}: {(np.median((phis.flatten() + np.pi - PHI_CRIT) % (2*np.pi)) + np.pi + PHI_CRIT) % (2*np.pi)}")
  plt.xlabel("φ [rad]")
  plt.ylabel("counts")
  plt.legend()
  plt.show()
  # phi hist
  for prednm in angles:
    phis, psis = angles[prednm]
    # cut off amino acids on the ends with only one angle:
    phis = phis[..., :-1]
    psis = psis[..., 1:]
    plt.scatter(phis.flatten(), psis.flatten(), marker=".", alpha=0.3, label=prednm[:20])
  plt.plot([-np.pi, np.pi], [PSI_CRIT, PSI_CRIT], linestyle="-", label="ψ₀", color="black")
  plt.plot([PHI_CRIT, PHI_CRIT], [-np.pi, np.pi], linestyle="-", label="φ₀", color="red")
  plt.legend()
  plt.xlim(-np.pi, np.pi)
  plt.ylim(-np.pi, np.pi)
  plt.xlabel("φ [rad]")
  plt.ylabel("ψ [rad]")
  plt.show()

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
    print(prednm)#plt.title(f"Ramachandran Plot - {prednm}")
    plt.xlabel("φ [rad]")
    plt.ylabel("ψ [rad]")
    plt.show()



def many_density_plot(angles, seq):
  for prednm in angles:
    L, batch, npep = angles[prednm][0].shape
    break
  plots_x, plots_y = approx_squarish_factorize(npep - 1)
  fig, axs = plt.subplots(plots_y, plots_x, figsize=(plots_x * 3, plots_y * 3), squeeze=False)
  axs_flat = axs.flatten()
  for prednm in angles:
    phis, psis = angles[prednm]
    # cut off amino acids on the ends with only one angle:
    phis = phis[..., :-1]
    psis = psis[..., 1:]
    for i in range(npep - 1):
      ax = axs_flat[i]
      ax.scatter(phis[:, :, i].flatten(), psis[:, :, i].flatten(), marker=".", alpha=0.3, label=prednm)
  for i in range(npep - 1):
    ax = axs_flat[i]
    ax.set_xlabel(f"amino {i + 1} ({seq[i + 1]}): φ")
    ax.set_ylabel(f"amino {i + 1} ({seq[i + 1]}): ψ")
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-np.pi, np.pi)
    #ax.legend()
  plt.tight_layout()
  plt.show()

def psi_hopping_plot(angles):
  for prednm in angles:
    phi, psi = angles[prednm]
    L, batch, npep = psi.shape
    delays = []
    for i in range(batch):
      for j in range(1, npep):
        psi_ij = psi[:, i, j]
        plt.scatter(np.arange(L), psi_ij, label=f"{prednm}: {j}", marker=".")
        regions = (-2. < psi_ij) & (psi_ij < 1.5)
        #plt.plot(regions)
        start = 0
        for k in range(1, L):
          if regions[k - 1] ^ regions[k]:
            delays.append(k - start)
            start = k
    print(prednm, delays)
  plt.xlabel("t [steps]")
  plt.ylabel("ψ [rad]")
  plt.legend()
  plt.show()

def phi_hopping_plot(angles):
  for prednm in angles:
    phi, psi = angles[prednm]
    phi = phi[:, :, :-1]
    psi = psi[:, :, 1:]
    L, must_be[1], npep = phi.shape
    fig, axs = plt.subplots(npep, 1, figsize=(10, 3*npep), sharex=True, gridspec_kw=dict(hspace=0))
    if npep == 1: axs = [axs] # put in a list in single case for consistency
    for k in range(npep):
      ax = axs[k]
      phi_k = phi[:, 0, k]
      psi_k = psi[:, 0, k]
      # Create scatter plot with phi as y-value and psi determining color
      ax.scatter(np.arange(L), phi_k,
        c=psi_k, cmap='hsv', vmin=-np.pi, vmax=np.pi,
        marker=".", alpha=0.4)
      ax.set_ylabel(f"φ {k + 1}")
      ax.set_ylim(-np.pi, np.pi)
      if k + 1 == npep:
        ax.set_xlabel("t [steps]")
    fig.suptitle(prednm)
    plt.show()

def angle_tcorr(angle, angle_crit, offset_max=None):
  """ angle: (L, batch, npep)
      angle_crit: float
      ans: (offset_max, npep) """
  L, batch, npep = angle.shape
  if offset_max is None: offset_max = L - 1
  sin_angles = np.sin(angle - angle_crit)
  print(sin_angles.mean((0, 1)))
  sin_angles = sin_angles - sin_angles.mean((0, 1))
  variance = (sin_angles**2).mean((0, 1))
  print(variance)
  return (np.stack([
    (sin_angles[offset:]*sin_angles[:L - offset]).mean((0, 1))
    for offset in range(offset_max)
  ])/variance)

def tcorr_plot(angles, offset_max):
  for prednm in angles:
    L, batch, npep = angles[prednm][0].shape
    break
  plots_x, plots_y = approx_squarish_factorize(2*(npep - 1))
  fig, axs = plt.subplots(plots_y, plots_x, figsize=(plots_x * 3, plots_y * 3), squeeze=False)
  axs_flat = axs.flatten()
  for prednm in angles:
    phis, psis = angles[prednm]
    # cut off amino acids on the ends with only one angle:
    phis = phis[..., :-1]
    psis = psis[..., 1:]
    must_be[L], must_be[batch], must_be[npep - 1] = phis.shape
    must_be[L], must_be[batch], must_be[npep - 1] = psis.shape
    tcorr_phis = angle_tcorr(phis, PHI_CRIT, offset_max)
    tcorr_psis = angle_tcorr(psis, PSI_CRIT, offset_max)
    for i in range(npep - 1):
      ax_phi = axs_flat[2*i]
      ax_psi = axs_flat[2*i + 1]
      for ax, tcorr in [(ax_phi, tcorr_phis), (ax_psi, tcorr_psis)]:
        ax.plot(tcorr[:, i], label=prednm)
  print(f"φ₀ = {PHI_CRIT}")
  print(f"ψ₀ = {PSI_CRIT}")
  for i in range(npep - 1):
    ax_phi = axs_flat[2*i]
    ax_psi = axs_flat[2*i + 1]
    ax_phi.set_title(f"amino {i + 1}: sin(φ - φ₀) covariance")
    ax_psi.set_title(f"amino {i + 1}: sin(ψ - ψ₀) covariance")
    ax_phi.set_ylabel("cov(sin(φ(t) - φ₀), sin(φ(t+nτ) - φ₀))")
    ax_psi.set_ylabel("cov(sin(ψ(t) - ψ₀), sin(ψ(t+nτ) - ψ₀))")
    ax_phi.set_ylim(-0.2, 1.)
    ax_psi.set_ylim(-0.2, 1.)
    for ax in [ax_phi, ax_psi]:
      ax.set_xlabel("offset, n [steps]")
      #ax.legend()
  plt.tight_layout()
  plt.show()


def f_alpha_beta(phis, psis):
  return np.sign(np.sin(psis - PSI_CRIT))*0.5*(1 - np.sign(np.sin(phis - PHI_CRIT)))

def f_tcorr(f, offset_max=None):
  """ f: (L, batch, npep)
      ans: (offset_max, npep) """
  if offset_max is None: offset_max = f.shape[0] - 1
  return (np.stack([
    (f**2).mean((0, 1))
  ] + [
    (f[offset:]*f[:-offset]).mean((0, 1))
    for offset in range(1, offset_max)
  ]))

def fcorr_plot(angles, offset_max):
  for prednm in angles:
    L, batch, npep = angles[prednm][0].shape
    break
  assert npep - 1 == 1, "don't have plotting figured out for more residues"
  for prednm in angles:
    phis, psis = angles[prednm]
    # cut off amino acids on the ends with only one angle:
    phis = phis[..., :-1]
    psis = psis[..., 1:]
    must_be[L], must_be[batch], must_be[npep - 1] = phis.shape
    must_be[L], must_be[batch], must_be[npep - 1] = psis.shape
    f = f_alpha_beta(phis, psis)
    tcorr = f_tcorr(f, offset_max=offset_max)
    plt.plot(tcorr, label=prednm)
  plt.legend()
  plt.ylim(0., 1.)
  plt.show()


def main(args):
  predictors = args_to_predictor_list(args)
  angles = {}
  for predictor in predictors:
    print(predictor.name)
    state = predictor.sample_q(args.batch)
    traj = predictor.predict(args.tmax, state)
    print(traj.size, traj.shape)
    phis, psis = get_angles(traj)
    angles[predictor.name] = (phis, psis)
    print("done")
  if args.plot_type == "density":
    density_plot(angles)
  elif args.plot_type == "tcorr":
    tcorr_plot(angles, args.offset_max)
  elif args.plot_type == "fcorr":
    fcorr_plot(angles, args.offset_max)
  elif args.plot_type == "manydensity":
    many_density_plot(angles, state.metadata.seq)
  elif args.plot_type == "psi_hopping":
    psi_hopping_plot(angles)
  elif args.plot_type == "phi_hopping":
    phi_hopping_plot(angles)
  elif args.plot_type == "heatmap":
    heatmap_plot(angles)



if __name__ == "__main__":
  from argparse import ArgumentParser
  parser = ArgumentParser(prog="test_model")
  add_model_list_arg(parser) # -M and -O
  parser.add_argument("--batch", type=int, default=1)
  parser.add_argument("--tmax", type=int, default=100)
  parser.add_argument("--plot_type", type=str, choices=["density", "tcorr", "fcorr", "manydensity", "psi_hopping", "phi_hopping", "heatmap"], default="density", help="Type of plot to generate")
  parser.add_argument("--offset_max", type=int, default=None)
  main(parser.parse_args())
