# Import necessary libraries
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import skimage
from skimage import segmentation, feature, future
from sklearn.ensemble import RandomForestClassifier
from functools import partial
import joblib

# Set the path for various directories
path = './slide/'

# Get a list of files in the path+'_colortiles/' directory
listimg = [item for item in os.listdir(path+'_colortiles/') if item.endswith('.png')]

A = []
B = []

# Feature Extraction
for i in range(len(listimg)):
    image = skimage.io.imread(path+'_colortiles/'+listimg[i])
    mask1 = skimage.io.imread(path+'_tiles/'+listimg[i].replace('.png','')+'_B.png')
    mask2 = skimage.io.imread(path+'_tiles/'+listimg[i].replace('.png','')+'_I.png')
    mask3 = skimage.io.imread(path+'_tiles/'+listimg[i].replace('.png','')+'_E.png')
    training_labels = np.zeros(image.shape[:2], dtype=np.uint8)

    indices = np.where(mask1 == [255])
    print(len(indices[0]))
    for j in range(len(indices[0])):
        training_labels[indices[0][j], indices[1][i]] = 1
    
    indices = np.where(mask2 == [255])
    print(len(indices[0]))
    for j in range(len(indices[0])):
        training_labels[indices[0][j], indices[1][i]] = 2

    indices = np.where(mask3 == [255])
    print(len(indices[0]))
    for j in range(len(indices[0])):
        training_labels[indices[0][j], indices[1][i]] = 3

    sigma_min = 1
    sigma_max = 16
    features_func = partial(feature.multiscale_basic_features,
                        intensity=True, edges=True, texture=True,
                        sigma_min=sigma_min, sigma_max=sigma_max,
                        channel_axis=-1)
    features = features_func(image)
    A.append(features)
    B.append(training_labels)

features = np.vstack(A)
training_labels = np.vstack(B)

# Features: save the array to a file
np.save('array_file_A.npy', A)

# Training_labels: save the array to a file
np.save('array_file_B.npy', B)

