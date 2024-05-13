import cv2
import numpy as np
import os

def loadImagesFromFolder(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)
            if img is not None:
                images.append(img)
    return images

def findAndDescribeFeatures(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(gray, None)
    return kp, des

def matchFeatures(des1, des2):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    return good_matches

def generateHomography(kp1, kp2, good_matches):
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    return H

def warpTwoImages(img1, img2, H):
    warp_img = cv2.warpPerspective(img2, H, (img1.shape[1]+img2.shape[1], img1.shape[0]))
    warp_img[0:img1.shape[0], 0:img1.shape[1]] = img1
    return warp_img

def multiStitching(images):
    img1, img2, img3 = images
    kp1, des1 = findAndDescribeFeatures(img1)
    kp2, des2 = findAndDescribeFeatures(img2)
    kp3, des3 = findAndDescribeFeatures(img3)
    good_matches1_2 = matchFeatures(des1, des2)
    good_matches2_3 = matchFeatures(des2, des3)
    H1_2 = generateHomography(kp1, kp2, good_matches1_2)
    H2_3 = generateHomography(kp2, kp3, good_matches2_3)
    warp_img = warpTwoImages(img1, img2, H1_2)
    warp_img = warpTwoImages(warp_img, img3, H2_3)
    return warp_img

def displayImages(img1, img2, img3, panorama):
    cv2.imshow('Image 1', img1)
    cv2.imshow('Image 2', img2)
    cv2.imshow('Image 3', img3)
    cv2.imshow('Panorama', panorama)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    folder_path = "E:\ComputerVision_OU\TH05\image2"
    images = loadImagesFromFolder(folder_path)
    
    if len(images) < 3:
        print("Error: Not enough images in the folder.")
    else:
        img1, img2, img3 = images[:3]  # Select the first three images
        panorama = multiStitching([img1, img2, img3])
        displayImages(img1, img2, img3, panorama)
