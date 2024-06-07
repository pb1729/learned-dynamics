import numpy as np
from scipy.optimize import linprog
import matplotlib.pyplot as plt
from matplotlib import colormaps


# UTILITIES FOR TASKS RELATED TO THE WASSERSTEIN METRIC
# mainly helpful for debugging


def xy_to_distances(X, Y):
  """ given X, Y that are lists of X and Y coordinates of points in the plane,
      output a distance matrix.
      X, Y: (n_pts)
      return: (n_pts, n_pts) """
  pts = np.stack([X, Y], axis=-1)
  displacements = pts[:, None] - pts[None, :]
  distances = np.linalg.norm(displacements, axis=-1)
  return distances


def compute_optimal_lipschitz(p1, p2, dist_matrix):
  # setup problem:
  X = len(p1)
  c = p1 - p2 # coefficients for the objective function
  A = []
  b = []
  for i in range(X):
    for j in range(X):
      if i != j:
        # f(i) - f(j) <= d(i, j)
        constraint = np.zeros(X)
        constraint[i] = 1
        constraint[j] = -1
        A.append(constraint)
        b.append(dist_matrix[i][j])
  A = np.array(A)
  b = np.array(b)
  # feed it to scipy to solve:
  result = linprog(c, A_ub=A, b_ub=b,
    bounds=(None, None)) # all vars in range (-inf, inf)
  if result.success:
    return result.x
  else:
    print(result)
    assert False, "didn't converge"


def show_hist(X, Y, p1, p2=None, dxdy=None):
  """ given a probability distribution p over the discrete set of points (X, Y)
      create a 3d histogram/box plot showing this distribution
      if two distributions p1, p2 are passed, we create a plot comparing them """
  if dxdy is None:
    x_deltas = abs(X[None, :] - X[:, None])
    dx = min(x_deltas[x_deltas > 0])
    y_deltas = abs(Y[None, :] - Y[:, None])
    dy = min(y_deltas[y_deltas > 0])
  else:
    dx, dy = dxdy
  fig = plt.figure()
  ax = fig.add_subplot(111, projection="3d")
  if p2 is None:
    p2 = np.zeros_like(p1)
  z0 = np.minimum(p1, p2)
  dz = abs(p1 - p2)
  soft_sign = ((p1 - p2) / (0.1/len(X) + dz))*0.5 + 0.5
  ax.bar3d(X, Y, z0, dx, dy, dz, color=colormaps["viridis"](soft_sign))
  plt.show()

def show_scatter(X, Y, *ps):
  fig = plt.figure()
  ax = fig.add_subplot(111, projection="3d")
  for p in ps:
    ax.scatter(X, Y, p)
  plt.show()



if __name__ == "__main__":
  X, Y = np.meshgrid(np.linspace(-1., 1., 20), np.linspace(-1., 1., 20))
  X, Y = X.flatten(), Y.flatten()
  distances = xy_to_distances(X, Y)
  p1 = np.exp(-1.1*(X**2 + Y**2))
  p2 = np.exp(-1.5*(X**2 + Y**2))
  p1 /= p1.sum()
  p2 /= p2.sum()
  f = compute_optimal_lipschitz(p1, p2, distances)
  show_hist(X, Y, p1, p2)
  show_scatter(X, Y, f, f)



