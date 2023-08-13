# Import necessary libraries
import numpy as np
import skimage
from skimage import segmentation, feature, future
from sklearn.ensemble import RandomForestClassifier
from functools import partial
import joblib


# Features: Load the array from the file
features = np.load('array_file_A.npy')

#Training_labels: Load the array from the file
training_labels = np.load('array_file_B.npy')


# Initialize the RandomForestClassifier with desired parameters
clf = RandomForestClassifier(n_estimators=50, n_jobs=-1,
                             max_depth=10, max_samples=0.05)

# Train the classifier using fit_segmenter
clf = future.fit_segmenter(training_labels, features, clf)

# Print the trained classifier
print(clf)

# Save the trained classifier to a file
joblib.dump(clf, "./random_forest.joblib")
