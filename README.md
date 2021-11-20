# Overview
Implementing camera calibration and structure from motion (SfM) methods to produce a reconstruction of a scene. Using the Direct Linear Transform (DLT) method to compute the scene geometry from correspondences.

## Setup

We recommend to use a virtual environment to avoid cluttering your Python installation. To set this up using the Python `venv` module on Linux just use
```
cd code
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
In case you want to use a different setup, just install the dependencies in requirements.txt manually.
```

## Data

The `data` folder contains all the data used for this project and is structured the following way:
```
`--data
   `--calibration
      |--image_size.txt
      |--points2d.txt
      |--points3d.txt
   `--images
   `--keypoints
   `--matches
```

### Calibration

The files contain informations about the 3D points of a scene and the 2D points of the scene seen from the camera. `image_size` only has two values for width and height of the image.

### Structure from motion

`images`: The different pictures taken from a moving camera. There are ten pictures with name `0000.png` to `0009.png`.

`keypoints`: All positions of the keypoints for each image. Each row of a file is a tuple which gives the position of the keypoint on the image:
```
`--keypoints
   |--0000.png.txt
   |-- ...
   |--0009.png.txt
```

`matches`: The matches between keypoints for a pair of images. For each pair (000i.png, 000j.png), `000i.png-000j.png.txt` contains the list of all keypoints and their corresponding index in the `keypoints` file of the images. For each keypoint, `000i.png-000j.png.txt` gives a pair (idx_i, idx_j) where idx_i is the index of the keypoint in 000i.png.txt and idx_j is the index of the keypoint in 000j.png.txt.
