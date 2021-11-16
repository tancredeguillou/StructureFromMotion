# StructureFromMotion
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

## Overview

The folder `code` contains the main code of this project while the data is in `data`.
