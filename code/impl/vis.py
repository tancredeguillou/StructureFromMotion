import matplotlib.pyplot as plt
import numpy as np

def Plot3DPoints(points, ax=None):
  if ax == None:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
  ax.plot(xs=points[:,0], ys=points[:,1], zs=points[:,2], color='k', marker='.', linestyle='None')
  ax.set_title('3D Scene')
  
  plt.show(block=False)

  return ax
  

def PlotCamera(R, t, ax=None, scale=1.0, color='b'):
  if ax == None:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

  Rcw = R.transpose()
  tcw = -Rcw @ t

  # Define a path along the camera gridlines
  camera_points = np.array([
    [0, 0, 0],
    [1, 1, 1],
    [1, -1, 1],
    [-1, -1, 1],
    [-1, 1, 1],
    [1, 1, 1],
    [0, 0, 0],
    [1, -1, 1],
    [0, 0, 0],
    [-1, -1, 1],
    [0, 0, 0],
    [-1, 1, 1]
  ])

  # Make sure that this vector has the right shape
  tcw = np.reshape(tcw, (3, 1))

  cam_points_world = (Rcw @ (scale * camera_points.transpose()) + np.tile(tcw, (1, 12))).transpose()

  ax.plot(xs=cam_points_world[:,0], ys=cam_points_world[:,1], zs=cam_points_world[:,2], color=color)

  plt.show(block=False)

  return ax


def Plot2DPoints(points, image_size, ax=None):
  if ax == None:
    fig = plt.figure()
    ax = fig.add_subplot(111)

  image_frame_points = np.array([
    [0, 0],
    [image_size[0], 0],
    [image_size[0], image_size[1]],
    [0, image_size[1]],
    [0, 0]
  ])
  ax.plot(image_frame_points[:,0], image_frame_points[:,1])
  ax.plot(points[:,0], image_size[1] - points[:,1], 'k.')
  ax.axis('equal')
  ax.set_title('Image')

  plt.show(block=False)


def PlotProjectedPoints(points3D, points2D, K, R, t, image_size, ax=None):
  if ax == None:
    fig = plt.figure()
    ax = fig.add_subplot(111)

  # Project points onto image
  p2d = K @ ((R @ points3D.transpose()) + t)
  p2d = p2d[0:2, :] / p2d[[-1],:]

  ax.plot(p2d[0,:], image_size[1] - p2d[1,:], 'r.')
  num_points = points2D.shape[0]
  for i in range(num_points):
    ax.plot([p2d[0,i], points2D[i,0]], [image_size[1] - p2d[1,i], image_size[1] - points2D[i,1]], color='g')

  plt.show(block=False)