from matplotlib.pyplot import axis
import numpy as np

def BuildProjectionConstraintMatrix(points2D, points3D):

  # For each correspondence, build the two rows of the constraint matrix and stack them
  num_corrs = points2D.shape[0]
  constraint_matrix = np.zeros((num_corrs * 2, 12))

  zeros = np.zeros((4,), dtype=int)

  for i in range(num_corrs):
    x = points2D[i]
    X = np.append(points3D[i], 1)
    first_row = np.concatenate((zeros, -X, x[1] * X), axis=None)
    second_row = np.concatenate((X, zeros, -x[0] * X), axis=None)
    constraint_matrix[2 * i] = first_row
    constraint_matrix[2 * i + 1] = second_row

  return constraint_matrix