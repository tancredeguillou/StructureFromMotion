import numpy as np

from impl.dlt import BuildProjectionConstraintMatrix
from impl.util import MakeHomogeneous, HNormalize
from impl.sfm.corrs import GetPairMatches
# from impl.opt import ImageResiduals, OptimizeProjectionMatrix

# # Debug
# import matplotlib.pyplot as plt
# from impl.vis import Plot3DPoints, PlotCamera, PlotProjectedPoints


def EstimateEssentialMatrix(K, im1, im2, matches):
  # Normalize coordinates (to points on the normalized image plane)
  H_kps1 = MakeHomogeneous(im1.kps, ax=1)
  H_kps2 = MakeHomogeneous(im2.kps, ax=1)

  # These are the keypoints on the normalized image plane (not to be confused with the normalization in the calibration exercise)
  normalized_kps1 = np.zeros(np.shape(H_kps1))
  normalized_kps2 = np.zeros(np.shape(H_kps2))
  for idx, kp in enumerate(H_kps1):
    normalized_kps1[idx] = np.linalg.inv(K) @ kp
  for idx, kp in enumerate(H_kps2):
    normalized_kps2[idx] = np.linalg.inv(K) @ kp

  # Assemble constraint matrix
  constraint_matrix = np.zeros((matches.shape[0], 9))

  for i in range(matches.shape[0]):
    # Add the constraints
    kp_1 = normalized_kps1[matches[i][0]]
    kp_2 = normalized_kps2[matches[i][1]]
    x1 = kp_1[0]
    y1 = kp_1[1]
    x2 = kp_2[0]
    y2 = kp_2[1]
    constraint_matrix[i] = np.array([x1*x2, x1*y2, x1, y1*x2, y1*y2, y1, x2, y2, 1])

  
  # Solve for the nullspace of the constraint matrix
  _, _, vh = np.linalg.svd(constraint_matrix)
  vectorized_E_hat = vh[-1,:]

  # Reshape the vectorized matrix to it's proper shape again
  E_hat = np.reshape(vectorized_E_hat, (3, 3))

  # We need to fulfill the internal constraints of E
  # The first two singular values need to be equal, the third one zero.
  # Since E is up to scale, we can choose the two equal singular values arbitrarily
  u, _, v = np.linalg.svd(E_hat)
  E = u @ np.diag([1, 1, 0]) @ v

  # This is just a quick test that should tell you if your estimated matrix is not correct
  # It might fail if you estimated E in the other direction (i.e. kp2' * E * kp1)
  # You can adapt it to your assumptions.
  for i in range(matches.shape[0]):
    kp1 = normalized_kps1[matches[i,0],:]
    kp2 = normalized_kps2[matches[i,1],:]

    assert(abs(kp1.transpose() @ E @ kp2) < 0.01)

  return E


def DecomposeEssentialMatrix(E):

  u, s, vh = np.linalg.svd(E)

  # Determine the translation up to sign
  t_hat = u[:,-1]

  W = np.array([
    [0, -1, 0],
    [1, 0, 0],
    [0, 0, 1]
  ])

  # Compute the two possible rotations
  R1 = u @ W @ vh
  R2 = u @ W.transpose() @ vh

  # Make sure the orthogonal matrices are proper rotations
  if np.linalg.det(R1) < 0:
    R1 *= -1

  if np.linalg.det(R2) < 0:
    R2 *= -1

  # Assemble the four possible solutions
  sols = [
    (R1, t_hat),
    (R2, t_hat),
    (R1, -t_hat),
    (R2, -t_hat)
  ]

  return sols

def TriangulatePoints(K, im1, im2, matches):

  R1, t1 = im1.Pose()
  R2, t2 = im2.Pose()
  P1 = K @ np.append(R1, np.expand_dims(t1, 1), 1)
  P2 = K @ np.append(R2, np.expand_dims(t2, 1), 1)

  # Ignore matches that already have a triangulated point
  new_matches = np.zeros((0, 2), dtype=int)

  num_matches = matches.shape[0]
  for i in range(num_matches):
    p3d_idx1 = im1.GetPoint3DIdx(matches[i, 0])
    p3d_idx2 = im2.GetPoint3DIdx(matches[i, 1])
    if p3d_idx1 == -1 and p3d_idx2 == -1:
      new_matches = np.append(new_matches, matches[[i]], 0)


  num_new_matches = new_matches.shape[0]

  points3D = np.zeros((num_new_matches, 3))

  for i in range(num_new_matches):

    kp1 = im1.kps[new_matches[i, 0], :]
    kp2 = im2.kps[new_matches[i, 1], :]

    # H & Z Sec. 12.2
    A = np.array([
      kp1[0] * P1[2] - P1[0],
      kp1[1] * P1[2] - P1[1],
      kp2[0] * P2[2] - P2[0],
      kp2[1] * P2[2] - P2[1]
    ])

    _, _, vh = np.linalg.svd(A)
    homogeneous_point = vh[-1]
    points3D[i] = homogeneous_point[:-1] / homogeneous_point[-1]


  # We need to keep track of the correspondences between image points and 3D points
  im1_corrs = new_matches[:,0]
  im2_corrs = new_matches[:,1]

  # Filter points behind the cameras by transforming them into each camera space and checking the depth (Z)
  # Make sure to also remove the corresponding rows in `im1_corrs` and `im2_corrs`
  cam1 = np.zeros(np.shape(points3D))
  cam2 = np.zeros(np.shape(points3D))
  #h_points_3D = MakeHomogeneous(points3D, ax=1)
  for i in range(np.shape(points3D)[0]):
    cam1[i] = R1 @ points3D[i] + t1
    cam2[i] = R2 @ points3D[i] + t2
  
  # Keep the Z values of the camera space points
  z1 = cam1[:, -1]
  z2 = cam2[:, -1]
  # We want the z values to be positive
  mask = np.all([z1 > 0, z2 > 0], axis=0)

  points3D = points3D[mask]
  im1_corrs = im1_corrs[mask]
  im2_corrs = im2_corrs[mask]

  return points3D, im1_corrs, im2_corrs

def EstimateImagePose(points2D, points3D, K):  

  # We use points in the normalized image plane.
  # This removes the 'K' factor from the projection matrix.
  # We don't normalize the 3D points here to keep the code simpler.
  H_points2D = MakeHomogeneous(points2D, ax=1)
  normalized_points2D = np.zeros(np.shape(H_points2D))
  for idx, H_point2D in enumerate(H_points2D):
    normalized_points2D[idx] = np.linalg.inv(K) @ H_point2D

  constraint_matrix = BuildProjectionConstraintMatrix(normalized_points2D, points3D)

  # We don't use optimization here since we would need to make sure to only optimize on the se(3) manifold
  # (the manifold of proper 3D poses). This is a bit too complicated right now.
  # Just DLT should give good enough results for this dataset.

  # Solve for the nullspace
  _, _, vh = np.linalg.svd(constraint_matrix)
  P_vec = vh[-1,:]
  P = np.reshape(P_vec, (3, 4), order='C')

  # Make sure we have a proper rotation
  u, s, vh = np.linalg.svd(P[:,:3])
  R = u @ vh

  if np.linalg.det(R) < 0:
    R *= -1

  _, _, vh = np.linalg.svd(P)
  C = np.copy(vh[-1,:])

  t = -R @ (C[:3] / C[3])

  return R, t

def TriangulateImage(K, image_name, images, registered_images, matches):
  # Loop over all registered images and triangulate new points with the new image.
  # Make sure to keep track of all new 2D-3D correspondences, also for the registered images
  image = images[image_name]
  points3D = np.zeros((0,3))
  # You can save the correspondences for each image in a dict and refer to the `local` new point indices here.
  # Afterwards you just add the index offset before adding the correspondences to the images.
  corrs = {}

  for registered_image_name in registered_images:
    registered_image = images[registered_image_name]
    pair_matches = GetPairMatches(registered_image_name, image_name, matches)
    t_points3D, im1_corrs, im2_corrs = TriangulatePoints(K, registered_image, image, pair_matches)
    points3D = np.append(points3D, t_points3D, 0)

    im1_dict = {}
    im2_dict = {}

    points3D_shape = np.shape(points3D)[0]
    t_points3D_shape = np.shape(t_points3D)[0]
    start_index = points3D_shape - t_points3D_shape
    stop_index = points3D_shape
    indexes = np.arange(start_index, stop_index)
    for i in range(t_points3D_shape):
      im1_dict[im1_corrs[i]] = indexes[i]
      im2_dict[im2_corrs[i]] = indexes[i]

    corrs[image_name] = im2_dict
    corrs[registered_image_name] = im1_dict

  return points3D, corrs
