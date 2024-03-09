import numpy as mp
import cv2
import numpy as np

#Define the camera matrix
fx = 800
fy = 800
cx = 640
cy = 480
cameraMatrix = np.array([[fx,0,cx],
                         [0,fy,cy],
                         [0,0,1]],np.float32)
#Define the rotation and translation vectors
rvec=np.zeros((3,1),np.float32)
tvec=np.zeros((3,1),np.float32)
#define the distortion coefficients
distCoeffs = np.ones((5,1),np.float32)
#define the 3D point in the wordld coordinate system
x,y,z = 10, 20, 30
points3D = np.array([[[x,y,z]]],np.float32)
#Map the 3D point to 2Dpoint
points2D, _ = cv2.projectPoints(points3D, rvec,tvec,cameraMatrix,distCoeffs)

#Display the 2D point
print ("2D Point:", points2D)