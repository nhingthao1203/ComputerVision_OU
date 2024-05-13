import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils.paths

def loadImages(path, resize):
    imagePath = list(imutils.paths.list_images(path))
    imagePath.sort()
    print(imagePath)
    listImage = []
    for i, j in enumerate(imagePath):
        image = cv2.imread(j)
        if resize == 1:
            image = cv2.resize(image, (int(image.shape[1] / 4), int(image.shape[0] / 4)))
        listImage.append(image)
    return listImage

def multiStitching(list_images):
    # Khởi tạo Sticher
    stitcher = cv2.Stitcher_create()
    # Tạo panorama
    status, panorama = stitcher.stitch(list_images)
    if status != cv2.Stitcher_OK:
        raise Exception("Stitching failed!")
    return panorama

# Load ảnh
list_images = loadImages('E:/ComputerVision_OU/TH05/image2', resize=0)

# Tạo panorama
panorama = multiStitching(list_images)

# Chuyển đổi sang dạng float và chuẩn hóa panorama
panorama = np.array(panorama, dtype=float) / 255
panorama = panorama[:, :, ::-1]  # Chuyển từ BGR sang RGB

# Hiển thị ảnh
f, axs = plt.subplots(1, len(list_images), sharey=True)
for idx, ax in enumerate(axs):
    ax.title.set_text(f"sequence {idx + 1}")
    ax.imshow(cv2.cvtColor(list_images[idx], cv2.COLOR_BGR2RGB))

f, ax = plt.subplots(1, 1, sharey=True)
ax.title.set_text(f"Result panorama")
ax.imshow(panorama)
plt.show()
