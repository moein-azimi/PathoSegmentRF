import numpy as np
import matplotlib.pyplot as plt
import skimage
from skimage import segmentation, feature, future
from sklearn.ensemble import RandomForestClassifier
from functools import partial
import joblib
import os
import cv2

# Define the path to the directory containing colorhist images
path = './slide/_colorhist/'

# Create a directory for RF masks if it doesn't exist
isExist = os.path.exists('./slide_output/')
if not isExist:
   os.mkdir('./slide_output/')
   
# Set the path for saving RF masks
path2 = './slide_output/'

# Get a list of all files ending with '.png' in the specified directory
x = [item for item in os.listdir(path) if item.endswith('.png')]

# Set parameters for multiscale feature extraction
sigma_min = 1
sigma_max = 16
features_func = partial(feature.multiscale_basic_features,
                        intensity=True, edges=True, texture=True,
                        sigma_min=sigma_min, sigma_max=sigma_max,
                        channel_axis=-1)

# Load the trained random forest model
loaded_rf = joblib.load("./random_forest.joblib")

# Iterate through each image in the directory
for i in range(len(x)):
    # Read the image
    image = skimage.io.imread(path+x[i])
    
    # Extract features using the defined features_func
    features = features_func(image)
    
    # Predict segmentation using the trained random forest model
    result = future.predict_segmenter(features, loaded_rf)

    # Create an image to store the mask
    img = np.zeros(result.shape[:2], dtype=np.uint8)

    # Set the pixels with label 2 to white
    indices = np.where(result == [2])
    for j in range(len(indices[0])):
        img[indices[0][j], indices[1][j]] = 255

    # Save the mask
    cv2.imwrite(path2+x[i].replace('.png','')+'_I.png', img)

    # Create an image to store the mask
    img = np.zeros(result.shape[:2], dtype=np.uint8)

    # Set the pixels with label 3 to white
    indices = np.where(result == [3])
    for j in range(len(indices[0])):
        img[indices[0][j], indices[1][j]] = 255

    # Save the mask
    cv2.imwrite(path2+x[i].replace('.png','')+'_E.png', img)
