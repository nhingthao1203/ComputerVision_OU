from imutils import paths
import imutils
import cv2
import numpy as np
import matplotlib.pyplot as plt

def loadImages(path, resize):
    imagePath = list(paths.list_images(path))
    imagePath.sort()
    print(imagePath)
    listImage = []
    for i, j in enumerate(imagePath):
        image = cv2.imread(j)
        if resize == 1:
            image = cv2.resize(image, (int(image.shape[1] / 4), int(image.shape[0] / 4)))
        listImage.append(image)
    return (listImage)

def findAndDescribeFeatures(image):
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    md = cv2.SIFT_create()

    keypoints, features = md.detectAndCompute(grayImage, None)

    features = np.float32(features)
    return keypoints, features

def matchFeatures(featuresA, featuresB, ratio=0.75):
    FLANN_INDEX_KDTREE = 0
    indexParams = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    searchParams = dict(checks=50)
    featureMatcher = cv2.FlannBasedMatcher(indexParams, searchParams)

    matches = featureMatcher.knnMatch(featuresA, featuresB, k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m, n in matches:
        if m.distance < ratio * n.distance:
            good.append(m)
    if len(good) > 4:
        return good
    else:
        raise Exception("Not enought matches")

def generateHomography(srcImg, dstImg, ransacRep=5.0):

    srcKP, srcFeatures = findAndDescribeFeatures(srcImg)
    dst_kp, dst_features = findAndDescribeFeatures(dstImg)

    good = matchFeatures(srcFeatures, dst_features)

    srcPoints = np.float32([srcKP[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dstPoints = np.float32([dst_kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(srcPoints, dstPoints, cv2.RANSAC, ransacRep)
    matchesMask = mask.ravel().tolist()
    return H, matchesMask

def crop(panorama, h_dst, conners):
    [xmin, ymin] = np.int32(conners.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(conners.max(axis=0).ravel() + 0.5)
    t = [-xmin, -ymin]
    conners = conners.astype(int)

    if conners[0][0][0] < 0:
        n = abs(-conners[1][0][0] + conners[0][0][0])
        panorama = panorama[t[1]:h_dst + t[1], n:, :]
    else:
        if (conners[2][0][0] < conners[3][0][0]):
            panorama = panorama[t[1]:h_dst + t[1], 0:conners[2][0][0], :]
        else:
            panorama = panorama[t[1]:h_dst + t[1], 0:conners[3][0][0], :]
    return panorama

def blendingMask(height, width, barrier, smoothingWindow, leftBiased=True):
    assert barrier < width
    mask = np.zeros((height, width))

    offset = int(smoothingWindow / 2)
    try:
        if leftBiased:
            mask[:, barrier - offset:barrier + offset + 1] = np.tile(np.linspace(1, 0, 2 * offset + 1).T,
                                                                     (height, 1))
            mask[:, :barrier - offset] = 1
        else:
            mask[:, barrier - offset:barrier + offset + 1] = np.tile(np.linspace(0, 1, 2 * offset + 1).T,
                                                                     (height, 1))
            mask[:, barrier + offset:] = 1
    except:
        if leftBiased:
            mask[:, barrier - offset:barrier + offset + 1] = np.tile(np.linspace(1, 0, 2 * offset).T, (height, 1))
            mask[:, :barrier - offset] = 1
        else:
            mask[:, barrier - offset:barrier + offset + 1] = np.tile(np.linspace(0, 1, 2 * offset).T, (height, 1))
            mask[:, barrier + offset:] = 1

    return cv2.merge([mask, mask, mask])

def panoramaBlending(dstImg_rz, srcImg_warped, width_dst, side, showstep=False):

    h, w, _ = dstImg_rz.shape
    smoothingWindow = int(width_dst / 8)
    barrier = width_dst - int(smoothingWindow / 2)
    mask1 = blendingMask(h, w, barrier, smoothingWindow=smoothingWindow, leftBiased=True)
    mask2 = blendingMask(h, w, barrier, smoothingWindow=smoothingWindow, leftBiased=False)

    if showstep:
        nonblend = srcImg_warped + dstImg_rz
    else:
        nonblend = None
        leftside = None
        rightside = None

    if side == 'left':
        dstImg_rz = cv2.flip(dstImg_rz, 1)
        srcImg_warped = cv2.flip(srcImg_warped, 1)
        dstImg_rz = (dstImg_rz * mask1)
        srcImg_warped = (srcImg_warped * mask2)
        pano = srcImg_warped + dstImg_rz
        pano = cv2.flip(pano, 1)
        if showstep:
            leftside = cv2.flip(srcImg_warped, 1)
            rightside = cv2.flip(dstImg_rz, 1)
    else:
        dstImg_rz = (dstImg_rz * mask1)
        srcImg_warped = (srcImg_warped * mask2)
        pano = srcImg_warped + dstImg_rz
        if showstep:
            leftside = dstImg_rz
            rightside = srcImg_warped

    return pano, nonblend, leftside, rightside

def warpTwoImages(srcImg, dstImg, showstep=False):

    H, _ = generateHomography(srcImg, dstImg)

    height_src, width_src = srcImg.shape[:2]
    height_dst, width_dst = dstImg.shape[:2]

    pts1 = np.float32([[0, 0], [0, height_src], [width_src, height_src], [width_src, 0]]).reshape(-1, 1, 2)
    pts2 = np.float32([[0, 0], [0, height_dst], [width_dst, height_dst], [width_dst, 0]]).reshape(-1, 1, 2)

    try:
        pts1_ = cv2.perspectiveTransform(pts1, H)
        pts = np.concatenate((pts1_, pts2), axis=0)

        [xmin, ymin] = np.int64(pts.min(axis=0).ravel() - 0.5)
        [xmax, ymax] = np.int64(pts.max(axis=0).ravel() + 0.5)
        t = [-xmin, -ymin]

        if (pts[0][0][0] < 0):
            side = 'left'
            width_pano = width_dst + t[0]
        else:
            width_pano = int(pts1_[3][0][0])
            side = 'right'
        height_pano = ymax - ymin

        Ht = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])
        srcImg_warped = cv2.warpPerspective(srcImg, Ht.dot(H), (width_pano, height_pano))

        dstImg_rz = np.zeros((height_pano, width_pano, 3))
        if side == 'left':
            dstImg_rz[t[1]:height_src + t[1], t[0]:width_dst + t[0]] = dstImg
        else:
            dstImg_rz[t[1]:height_src + t[1], :width_dst] = dstImg

        pano, nonblend, leftside, rightside = panoramaBlending(dstImg_rz, srcImg_warped, width_dst, side,
                                                                      showstep=showstep)

        pano = crop(pano, height_dst, pts)
        return pano, nonblend, leftside, rightside
    except:
        raise Exception("Please try again with another image set!")

def multiStitching(list_images):
    n = int(len(list_images) / 2 + 0.5)
    left = list_images[:n]
    right = list_images[n - 1:]
    right.reverse()
    while len(left) > 1:
        dstImg = left.pop()
        srcImg = left.pop()
        left_pano, _, _, _ = warpTwoImages(srcImg, dstImg)
        left_pano = left_pano.astype('uint8')
        left.append(left_pano)

    while len(right) > 1:
        dstImg = right.pop()
        srcImg = right.pop()
        right_pano, _, _, _ = warpTwoImages(srcImg, dstImg)
        right_pano = right_pano.astype('uint8')
        right.append(right_pano)

    if (right_pano.shape[1] >= left_pano.shape[1]):
        fullpano, _, _, _ = warpTwoImages(left_pano, right_pano)
    else:
        fullpano, _, _, _ = warpTwoImages(right_pano, left_pano)
    return fullpano

list_images = loadImages('E:/ComputerVision_OU/TH05/image2', 0)

panorama= multiStitching(list_images)
panorama = np.array(panorama,dtype=float)/float(255)
panorama = panorama[:,:,::-1]

f, axs = plt.subplots(1, len(list_images), sharey=True)
for idx, ax in enumerate(axs):
    ax.title.set_text(f"sequence {idx + 1}")
    ax.imshow(list_images[idx][:,:,::-1])

f, ax = plt.subplots(1, 1, sharey=True)
ax.title.set_text(f"Result panorama")
ax.imshow(panorama)