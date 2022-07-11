# encoding: utf-8
import cv2
import numpy as np


def normalization(data):
  '''This function normalize image data that has 255 pixels

  Input
  -----
  data :
    The data to be normalized

  Output
  ------
  normalized_data :
    The data normalized
  '''

  normalized_data = data / 255
  return normalized_data

def to_gray(image_data):
  '''This function change an RGB image to grayscale

  Input
  -----
  image_data :
    The RGB image data/pixels

  Output
  ------
  gray_image :
    The grayscale image (pixels)
  '''

  samples = image_data.shape[0]
  x_size = image_data.shape[1]
  y_size = image_data.shape[2]

  gray_image = np.zeros(shape=(samples,x_size,y_size))

  for i, image in enumerate(image_data):
      gray_image[i,:,:] = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  return gray_image
