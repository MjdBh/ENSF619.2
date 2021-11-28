
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MjdBh/ENSF619.2/blob/main/Assignment-03/assignment03.ipynb)
# This assignment is for the course 619 at the University of Calgary.
##First, we will remove the blank pixels from the image.
### We will use bounding boxes to remove the blank pixels.

##Second, we will apply multiple augmentations to the image.
### We will use the following augmentations:
 * rotation_range
 * width_shift_range
 * brightness_range
 * height_shift_range
 * shear_range
 * horizontal_flip
 * vertical_flip
 * zoom_range
 * fill_mode

All of them parametrized with random values, and in every iteration we will get multiple of them.