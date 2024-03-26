import numpy as np
import cv2

roi = cv2.imread('ball.png')
hsvr = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
#target is the image we search in
target = cv2.imread('grass.png')

#Step 1 -calculate hist for 2 chanels H,S
hsvt = cv2.cvtColor(target,cv2.COLOR_BGR2HSV)
#Find the histograms using calHist
M=cv2.calcHist([hsvr],[0,1],None,[180,256],[0,180,0,256])
I = cv2.calcHist([hsvt],[0,1],None,[180,256],[0,180,0,256])

#Step 2-Find the ratio R = M/I
R = M/(I+1)
#Step 3 - Backproject R
h,s,v= cv2.split(hsvt)
B = R[h.ravel(), s.ravel()]
B = np.minimum(B,1)
B = B.reshape(hsvt.shape[:2])

#Step 4- convolution with a circular disc, B = D*B
disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
cv2.filter2D(B,-1,disc,B)
B = np.uint8(B)
cv2.normalize(B,B,0,255,cv2.NORM_MINMAX)
#Step 5-thresholding for a suitable value give a nice result
ret,thresh = cv2.threshold(B,10,255,0)
#Overlay images using bitwise_and
thresh = cv2.merge((thresh,thresh,thresh))
res= cv2.bitwise_and(target,thresh)

#Display  the output
res = np.vstack((target,thresh,res))
cv2.imshow('Result',res)
cv2.waitKey(0)