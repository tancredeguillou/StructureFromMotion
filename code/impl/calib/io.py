import numpy as np
import os.path

def ReadPoints2D(data_folder):
  points2d_path = os.path.join(data_folder, 'calibration', 'points2d.txt')
  points2d = np.loadtxt(points2d_path)
  return points2d


def ReadPoints3D(data_folder):
  points3d_path = os.path.join(data_folder, 'calibration', 'points3d.txt')
  points3d = np.loadtxt(points3d_path)
  return points3d


def ReadImageSize(data_folder):
  im_size_path = os.path.join(data_folder, 'calibration', 'image_size.txt')
  im_size = np.loadtxt(im_size_path, dtype=int)
  return im_size