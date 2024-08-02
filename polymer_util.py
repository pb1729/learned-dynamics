import numpy as np
import torch

from sims import get_poly_tc
from utils import must_be


def sinhc(x):
  """ sinhc(x) = sinh(x)/x
      hyperbolic analog of the sinc function, though without scaling by π.
      product approximation used here is a truncation of (1080) in Jolley's Summation of Series """
  ans = 1.
  z = x.copy()
  for _ in range(5):
    z *= 0.5
    ans *= np.cosh(z)
  return ans

def rouse(n, length):
  """ get nth Rouse mode for polymer of a given length """
  return np.cos(n*np.pi*(0.5 + np.arange(length))/length) - 0.5*(n == 0)

def rouse_k(n, k, length):
  """ get the mode spring constant [/TT] for nth Rouse mode for polymer of given length
      and bond spring constant k [/TT]. """
  return 4*k*(np.sin(0.5*np.pi*n/length))**2

def tica_theory(sim):
  n = np.arange(1, sim.poly_len)
  return np.exp(-sim.delta_t/get_poly_tc(sim, rouse_k(n, sim.k, sim.poly_len)))

def rouse_block(length):
  """ get the entire block of rouse mode coefficients for a given polymer length
      ans: (length, length) --> ans[i_atom, n_mode] """
  return np.stack([
    rouse(n, length)
    for n in range(length)
  ], axis=-1)

def rouse_block_unitary(length):
  """ get the entire block of rouse mode coefficients, normalized to give a unitary matrix
      ans: (length, length) --> ans[i_atom, n_mode] """
  n = np.arange(0, length)[:, None]
  m = np.arange(0, length)[None, :]
  return np.cos(m*np.pi*(0.5 + n)/length)*np.sqrt((2. - (m == 0))/length)

class RouseEvolver:
  def __init__(self, config, device="cuda"):
    sim = config.sim
    length = sim.poly_len
    drag = get_poly_tc(sim, 1.)
    mode_k = rouse_k(np.arange(0, length), sim.k, length)
    quickness = sim.delta_t/drag
    self.sigmas = np.sqrt(np.exp(-quickness*mode_k)*2*quickness*sinhc(quickness*mode_k))
    self.decays = np.exp(-quickness*mode_k)
    self.rouse_block = rouse_block_unitary(length)
    # now move everything to a torch tensor on the device
    self.sigmas = torch.tensor(self.sigmas, device=device, dtype=torch.float32)
    self.decays = torch.tensor(self.decays, device=device, dtype=torch.float32)
    self.rouse_block = torch.tensor(self.rouse_block, device=device, dtype=torch.float32)
  def predict(self, x0, noise):
    """ sample x1 given x0 for a Rouse chain with the same parameters as this sim
        noise is expected to be randn() with same shape as x0 """
    batch, length, space_dim = x0.shape
    must_be[batch], must_be[length], must_be[space_dim] = noise.shape
    modes = torch.einsum("bnv, nm -> bmv", x0, self.rouse_block)
    modes *= self.decays[..., None]
    modes += self.sigmas[..., None]*noise
    return torch.einsum("bmv, nm -> bnv", modes, self.rouse_block)


# ARBITRARY QUANTA THEORY (THE IDEALISTIC THEORY)
def get_log_expected_singular_values(log_vals, n, show_ans_tups=False):
    """ get the expected singular values where arbitrary numbers
        of quanta can be put into a given mode
        log_vals: logarithms of the value for 1 quanta in a given mode.
                  Should be sorted in increasing order!
        n: we'll compute the largest (least negative) n singular values """
    def H(tup): # energy function
        return sum([tup[i]*log_vals[i] for i in range(dim)])
    dim = len(log_vals)
    ans_tups = [tuple([0]*dim)]
    for i in range(n - 1):
        # get a list of candidates for the next largest tuple
        # a candidate must not be an existing tuple
        # all candidates are increments of some existing tuple
        candidate_tups = []
        for tup in ans_tups:
            for j in range(dim):
                lst = list(tup)
                lst[j] += 1
                inc_tup = tuple(lst)
                if inc_tup not in ans_tups:
                    candidate_tups.append(inc_tup)
                    break
        # find and add to the list the largest of the candidates
        i_best = 0
        H_best = -np.inf
        for i, tup in enumerate(candidate_tups):
            if H(tup) > H_best:
                i_best = i
                H_best = H(tup)
        ans_tups.append(candidate_tups[i_best])
    if show_ans_tups:
        for tup in ans_tups: print(tup)
    return [H(tup) for tup in ans_tups]

def get_n_quanta_theory(num_values, sim):
    log_lin_S = np.log(tica_theory(sim))
    return np.exp(get_log_expected_singular_values(log_lin_S, num_values))



if __name__ == "__main__":
  poly_len = 12
  # test functionality by plotting the rouse modes...
  import matplotlib.pyplot as plt
  x = np.arange(0, poly_len)
  for n in range(poly_len):
    plt.scatter(x, rouse(n, poly_len))
    plt.show()



