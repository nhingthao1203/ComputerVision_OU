import cv2
import numpy as np
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
# Read the image
image_path = "E:/ComputerVision_OU/TH06ver2/OIP.jpg"
frame = cv2.imread(image_path)
frame = cv2.resize(frame, (640, 480)) # Resize the image if needed
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# Detect pedestrians
boxes, weights = hog.detectMultiScale(frame, winStride=(8, 8))
# Draw bounding boxes
for (x, y, w, h) in boxes:
cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
# Display the image with bounding boxes
cv2.imshow('frame', frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
