import numpy as np

from impl.dlt import BuildProjectionConstraintMatrix
from impl.util import HNormalize

def NormalizePoints3D(points):
  
  # Compute the center and spread of points
  center = np.mean(points, 0)
  offsets = points - np.tile(center, (points.shape[0], 1))
  dists = np.linalg.norm(offsets, axis=1)

  T_inv = np.eye(4) * np.mean(dists)
  T_inv[3,3] = 1
  T_inv[0:3,3] = center

  # Invert this so that after the transformation, the points are centered and their mean distance to the origin is 1
  T = np.linalg.inv(T_inv)

  # Normalize the points
  normalized_points3D = (T @ np.append(points, np.ones((points.shape[0], 1)), 1).transpose()).transpose()

  return normalized_points3D[:,0:3], T


def NormalizePoints2D(points, image_size):
  # Assume the image spans the range [-1, 1] in both dimensions and normalize the points accordingly
  T_inv = np.eye(3)
  T_inv[0,0] = image_size[0] / 2
  T_inv[1,1] = image_size[1] / 2
  T_inv[0,2] = image_size[0] / 2
  T_inv[1,2] = image_size[1] / 2

  T = np.linalg.inv(T_inv)

  normalized_points2D = (T @ np.append(points, np.ones((points.shape[0], 1)), 1).transpose()).transpose()

  return normalized_points2D[:,0:2], T


def EstimateProjectionMatrix(points2D, points3D):
  
  # TODO Build constraint matrix
  # Hint: Pay attention to the assumed order of the vectorized P matrix. You will need the same order when rehaping the vector to the matrix later
  constraint_matrix = BuildProjectionConstraintMatrix(points2D, points3D)

  # Solve for the nullspace
  _, _, vh = np.linalg.svd(constraint_matrix)
  P_vec = vh[-1,:]

  # TODO: Reshape the vector to a matrix (pay attention to the order)
  P = 

  return P


def DecomposeP(P):
  # TODO
  # Decompose P into K, R, and t

  # P = K[R|t] = K[R | -RC] = [KR | -KRC]
  # We could decompose KR with a RQ decomposition since K is upper triangular and R is orthogonal
  # To switch this around we set M = KR -> M^(-1) = R^(-1) K^(-1) and can use the QR decomposition on M^(-1)

  # TODO
  # Find K and R
  K =
  R =


  # TODO
  # It is possible that a sign was assigned to the wrong matrix during decomposition
  # We need to make sure that det(R) = 1 to have a proper rotation
  # We also want K to have a positive diagonal


  # TODO
  # Find the camera center C as the nullspace of P
  C = 

  # TODO
  # Compute t from R and C
  t = 

  return K, R, t