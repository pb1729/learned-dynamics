import sympy as sp
from sympy import jacobi, Poly
import torch
import torch.nn.functional as F


x = sp.symbols("x")
def shifted_jacobi(n, a, b):
  """ Jacobi polynomials on the interval [0, 1] instead of [-1, 1] """
  return Poly(jacobi(n, a, b, (2*x - 1)), x, domain="R")


def jacobi_matrix(dim):
  polynomials = [shifted_jacobi(n, 1, 0) for n in range(dim)]
  # Extract coefficients from the polynomials
  A = torch.zeros(dim, 1 + dim, device="cuda")
  for i, poly in enumerate(polynomials):
    coeffs = poly.all_coeffs()[::-1]
    for j in range(len(coeffs)):
      A[i, j] = float(coeffs[j])
  return A


def get_radial_encode(dim):
  A = jacobi_matrix(dim)
  # TODO: should we torch.compile this?
  def radial_encode(r, rmax):
    """ r: (..., 3)
        ans: (..., dim) """
    z = (torch.linalg.vector_norm(r, dim=-1)/rmax)[..., None]
    pows = z**torch.arange(A.shape[1], device="cuda")
    return F.linear(pows, A)*torch.relu(1. - z)
  return radial_encode


# define the functions we'll be using...
radial_encode_8 = get_radial_encode(8)


if __name__ == "__main__":
  import matplotlib.pyplot as plt
  r = torch.linspace(-5., 5., 680, device="cuda")
  y = radial_encode_8(r[:, None], 4.4)
  plt.plot(r.cpu(), y.cpu())
  plt.show()
