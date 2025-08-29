import cuequivariance as cueq

from .utils import must_be


def so3_c(l, r, o):
  ans = cueq.clebsch_gordan(cueq.SO3(l), cueq.SO3(r), cueq.SO3(o))
  num_solutions, must_be[2*l + 1], must_be[2*r + 1], must_be[2*o + 1] = ans.shape
  assert num_solutions > 0, f"{l}, {r} -> {o} tensor product does not exist!"
  assert num_solutions <= 1, f"{l}, {r} -> {o} tensor product is ambiguous!"
  return ans.squeeze(0)

def add_tens_prods(rank, tens_prod, l_segs, r_segs, tp_list):
  """ MUTATES tens_prod by adding output segments and paths for each tensor prod
      returns the irreps of the output """
  o_segs = []
  o_irreps = []
  for l, r, o in tp_list:
    o_segs.append(tens_prod.add_segment(2, (2*o + 1, rank))) # add appropriate output segment (operand 2 is the output)
    o_irreps.append((rank, o))
  for i, (l, r, o) in enumerate(tp_list):
    c = so3_c(l, r, o)
    tens_prod.add_path(l_segs[l], r_segs[r], o_segs[i], c=c)
  return cueq.Irreps(cueq.SO3, o_irreps)

def make_elementwise_tensor_product(rank, prods, l_irreps=(0, 1), r_irreps=(0, 1)):
  tens_prod = cueq.SegmentedTensorProduct.from_subscripts("iu,ju,ku+ijk")
  l_irreps = [
    tens_prod.add_segment(0, (2*irrep + 1, rank))
    for irrep in l_irreps]
  r_irreps = [
    tens_prod.add_segment(1, (2*irrep + 1, rank))
    for irrep in r_irreps
  ]
  output_irreps = add_tens_prods(
    rank, tens_prod,
    l_irreps, r_irreps,
    prods)
  # API will only accept operations on segments that are 0-D and 1-D, break irrep dims into individual segments
  tens_prod = tens_prod.flatten_modes(["i", "j", "k"])
  poly = cueq.SegmentedPolynomial.eval_last_operand(tens_prod) # last operand (operand 2) is the output
  return poly, output_irreps

def make_spherical_harmonics_tensor_product(out_irreps):
  tens_prod = cueq.descriptors.spherical_harmonics(cueq.SO3(1), out_irreps)
  tens_prod = tens_prod.polynomial # get the SegmentedPolynomial from inside the EquivariantPolynomial
  tens_prod = tens_prod.flatten_coefficient_modes()
  return tens_prod
