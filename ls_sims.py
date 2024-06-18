from sims import sims, get_poly_tc
from polymer_util import rouse_k


if __name__ == "__main__":
  print("showing the size of delta_t measured in terms of Rouse times for all 3d_ou_poly sims")
  for simnm in sims:
    if simnm[:10] == "3d_ou_poly":
      sim = sims[simnm]
      Δt = sim.delta_t/get_poly_tc(sim, rouse_k(1, sim.k, sim.poly_len))
      print("%s:\t  Δt = %f rouse times" % (simnm, Δt))



