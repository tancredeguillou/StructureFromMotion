import os
import matplotlib.pyplot as plt
import numpy as np

# Simple image class to hold some information for the SfM task.
class Image:
  # Constructor that reads the keypoints and image file from the data directory
  def __init__(self, data_folder, name):
      image_path = os.path.join(data_folder, 'images', name)
      keypoints_path = os.path.join(data_folder,'keypoints', name + '.txt')

      self.name = name
      self.image = plt.imread(image_path)
      self.kps = np.loadtxt(keypoints_path)
      self.p3D_idxs = {}

  # Set the image pose.
  # This is assumed to be the transformation from global space to image space
  def SetPose(self, R, t):
    self.R = R
    self.t = t

  # Get the image pose
  def Pose(self):
    return self.R, self.t

  # Add a new 2D-3D correspondence to the image
  # The function expects two equal length lists of indices, the first one with the
  # keypoint index in the image, the second one with the 3D point index in the reconstruction.
  def Add3DCorrs(self, kp_idxs, p3D_idxs):
    for corr in zip(kp_idxs, p3D_idxs):
      self.p3D_idxs[corr[0]] = corr[1]

  # Get the 3D point associated with the given keypoint.
  # Will return -1 if no correspondence is set for this keypoint.
  def GetPoint3DIdx(self, kp_idx):
    if kp_idx in self.p3D_idxs:
      return self.p3D_idxs[kp_idx]
    else:
      return -1

  # Get the number of 3D points that are observed by this image
  def NumObserved(self):
    return len(self.p3D_idxs)