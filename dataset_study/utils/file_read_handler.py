# encoding: utf-8
import csv
import pickle

import numpy as np


def read_rgb_dataset(filepath):
    with open(filepath, 'rb') as f:
        file_dict = pickle.load(f, encoding='latin1')  # dictionary type, we use 'latin1' for python3

        x = file_dict['features'].astype(np.float32)   # 4D numpy.ndarray type, for train = (34799, 32, 32, 3)
        y = file_dict['labels']                        # 1D numpy.ndarray type, for train = (34799,)
        s = file_dict['sizes']                         # 2D numpy.ndarray type, for train = (34799, 2)
        c = file_dict['coords']                        # 2D numpy.ndarray type, for train = (34799, 4)
        """
        Data is a dictionary with four keys:
            'features' - is a 4D array with raw pixel data of the traffic sign images,
                         (number of examples, width, height, channels).
            'labels'   - is a 1D array containing the label id of the traffic sign image,
                         file label_names.csv contains id -> name mappings.
            'sizes'    - is a 2D array containing arrays (width, height),
                         representing the original width and height of the image.
            'coords'   - is a 2D array containing arrays (x1, y1, x2, y2),
                         representing coordinates of a bounding frame around the image.
        """

    return x, y, s, c

def read_csv(filepath):
    label_list = []
    with open(filepath, 'r') as file:
        reader = csv.reader(file)

        # move the reader object to point on the next row to exclude header
        next(reader)

        for row in reader:
            label_list.append(row[1])
        
    return label_list
