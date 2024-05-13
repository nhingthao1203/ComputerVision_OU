import os
import glob
import cv2
from skimage.feature import hog
from sklearn.externals import joblib
from skimage.transform import pyramid_gaussian
from skimage import color 

from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import LabelEncoder
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.metrics import classification_report

from imutils.object_detection import non_max_suppression
import imutils

# Training SVM

X = []
Y = []

pos_im_path = 'inria-person/data_ped/no_pedestrians'
neg_im_path = 'inria-person/data_ped/pedestrians'

# Load the positive features
for filename in glob.glob(os.path.join(pos_im_path, '*.png')):
    fd = cv2.imread(filename, 0)
    fd = cv2.resize(fd, (64, 128))
    fd = hog(fd, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys', visualise=False, transform_sqrt=True, feature_vector=True)
    X.append(fd)
    Y.append(1)

# Load the negative features
for filename in glob.glob(os.path.join(neg_im_path, '*.png')):
    fd = cv2.imread(filename, 0)
    fd = cv2.resize(fd, (64, 128))
    fd = hog(fd, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys', visualise=False, transform_sqrt=True, feature_vector=True)
    X.append(fd)
    Y.append(0)

X = np.float32(X)
Y = np.array(Y)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

print('Train data:', len(X_train))
print('Train label:(1,0)', len(y_train))

model = LinearSVC()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Confusion matrix and accuracy

print('Classification report for classifier {model}:\n'
      f"{metrics.classification_report(y_test, y_pred)}\n")

joblib.dump(model, 'models.dat')
print('Model saved: {}'.format('models.dat'))

# Pedestrian detection

modelFile ='models.dat'
inputFile = 'inria-person/data_ped/no_pedestrians/01-03d.jpg'
outFile = 'output.jpg'
image = cv2.imread(inputFile)
image = cv2.resize(image,(400,256))
size =(64,128)
step_size=(9,9)
downscale=1.05

# List to store the detections
detections = []

# Load the SVM model
model = joblib.load(modelFile)

def sliding_window(image, window_size, step_size):
    for y in range(0, image.shape[0], step_size[1]):
        for x in range(0, image.shape[1], step_size[0]):
            yield (x, y, image[y: y + window_size[1], x: x + window_size[0]])

def pedestrian_detection(image, model, size=(64, 128), step_size=(9, 9), downscale=1.05):
    detections = []
    for image_scale in pyramid_gaussian(image, downscale=downscale):
        if image_scale.shape[0] < size[1] or image_scale.shape[1] < size[0]:
            break
        for (x, y, window) in sliding_window(image_scale, size, step_size):
            if window.shape[0] != size[1] or window.shape[1] != size[0]:
                continue
            window = color.rgb2gray(window)
            fd = hog(window, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys', visualize=False, transform_sqrt=True, feature_vector=True)
            fd = np.array(fd).reshape(1, -1)
            pred = model.predict(fd)
            if pred == 1:
                if model.decision_function(fd) > 0.5:
                    detections.append((int(x * (downscale**scale)), int(y * (downscale**scale)), model.decision_function(fd), int(size[0] * (downscale**scale)), int(size[1] * (downscale**scale))))
    return detections

detections = pedestrian_detection(image, model)

# Draw rectangles around detected pedestrians
for (x, y, _, w, h) in detections:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Save the output image
cv2.imwrite(outFile, image)
