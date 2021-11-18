from matplotlib.pyplot import draw
import numpy as np
import scipy as sp

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
  
  constraint_matrix = BuildProjectionConstraintMatrix(points2D, points3D)

  # Solve for the nullspace
  _, _, vh = np.linalg.svd(constraint_matrix)
  P_vec = vh[-1,:]
  
  P = np.reshape(P_vec, (3, 4))

  return P


def DecomposeP(P):
  # Decompose P into K, R, and t

  # P = K[R|t] = K[R | -RC] = [KR | -KRC]
  # We could decompose KR with a RQ decomposition since K is upper triangular and R is orthogonal
  # To switch this around we set M = KR -> M^(-1) = R^(-1) K^(-1) and can use the QR decomposition on M^(-1)

  # M = KR is a 3x3 matrix
  M = P[:,:3]
  R_inv, K_inv = np.linalg.qr(np.linalg.inv(M))

  # Find K and R
  K_hat = np.linalg.inv(K_inv)
  R_hat = np.linalg.inv(R_inv)
  # It is possible that a sign was assigned to the wrong matrix during decomposition
  # We need to make sure that det(R) = 1 to have a proper rotation
  # We also want K to have a positive diagonal
  T = np.diag(np.sign(np.diag(K_hat)))
  K = K_hat @ T
  R_hat = np.linalg.inv(T) @ R_hat
  #R = HNormalize(R_hat) if linalg.det(R_hat) == 1 else -HNormalize(R_hat)
  R = R_hat if np.linalg.det(R_hat) > 0 else -R_hat
  assert(np.linalg.det(R) > 0)

  # Find the camera center C as the nullspace of P
  C = HNormalize(sp.linalg.null_space(P))

  # Compute t from R and C
  t = -(R @ C)

  return K, R, t