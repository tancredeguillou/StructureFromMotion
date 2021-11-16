import math
import matplotlib.pyplot as plt
import numpy as np

from impl.vis import PlotCamera

def PlotImages(images):
  num_images = len(images)

  grid_height = math.floor(math.sqrt(num_images))
  grid_width = math.ceil(num_images / grid_height)

  fig = plt.figure()

  for idx, image_name in enumerate(images):
    ax = fig.add_subplot(grid_height, grid_width, idx+1)
    ax.imshow(images[image_name].image)
    ax.axis('off')

  plt.show(block=False)


def PlotWithKeypoints(image):
  fig = plt.figure()
  ax = fig.add_subplot(1,1,1)
  ax.imshow(image.image)
  ax.plot(image.kps[:,0], image.kps[:,1], 'r.')
  ax.axis('off')
  plt.show(block=False)


def PlotImagePairMatches(im1, im2, matches):
  pair_image_width = im1.image.shape[1] + im2.image.shape[1]
  pair_image_height = max(im1.image.shape[0], im2.image.shape[0])

  pair_image = np.ones((pair_image_height, pair_image_width, 3))

  im2_offset = im1.image.shape[1]

  pair_image[0:im1.image.shape[0], 0:im1.image.shape[1], :] = im1.image
  pair_image[0:im2.image.shape[0], im2_offset : im2.image.shape[1] + im2_offset] = im2.image

  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.imshow(pair_image)
  ax.plot(im1.kps[:,0], im1.kps[:,1], 'r.')
  ax.plot(im2.kps[:,0] + im2_offset, im2.kps[:,1], 'r.')
  for i in range(matches.shape[0]):
    kp1 = im1.kps[matches[i,0]]
    kp2 = im2.kps[matches[i,1]] 
    ax.plot([kp1[0], kp2[0] + im2_offset], [kp1[1], kp2[1]], 'g-', linewidth=1)
  ax.axis('off')
  ax.set_title(f'{im1.name} - {im2.name} ({matches.shape[0]})')
  plt.show()
  plt.close(fig)


def PlotCameras(images, registered_images, ax=None):

  for image_name in registered_images:
    image = images[image_name]
    R, t = image.Pose()
    ax = PlotCamera(R, t, ax, 0.5)