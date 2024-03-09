import numpy as np
import cv2
import matplotlib.pyplot as plt

def pinholeCamera(imageSize=(400, 400), pinholeSize=(3, 3), objectPosition=(200, 200), objectSize=(20, 100)):
    # Create a black image
    image = np.zeros((imageSize[0], imageSize[1], 3), dtype=np.uint8)

    # Simulate pinhole by setting a region to white where light passes through
    pinholeStart = (objectPosition[0] - pinholeSize[0] // 2, objectPosition[1] - pinholeSize[1] // 2)
    pinholeEnd = (pinholeStart[0] + pinholeSize[0], pinholeStart[1] + pinholeSize[1])
    image[pinholeStart[1]:pinholeEnd[1], pinholeStart[0]:pinholeEnd[0]] = [255, 255, 255]

    # Simulate the object in the scene
    candleStart = (objectPosition[0] - objectSize[0] // 2, objectPosition[1] - objectSize[1] // 2)
    candleEnd = (candleStart[0] + objectSize[0], candleStart[1] + objectSize[1])
    image[candleStart[1]:candleEnd[1], candleStart[0]:candleEnd[0]] = [0, 165, 255]  # color for the object

    return image

def plotImages(images, titles):
    fig, axes = plt.subplots(1, len(images), figsize=(12, 4))
    for ax, image, title in zip(axes, images, titles):
        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for matplotlib
        ax.set_title(title)
        ax.axis('off')
    plt.show()

# Simulate pinhole camera image formation with a candle object
pinholeSize = (10, 10)

I1 = pinholeCamera((400, 400), pinholeSize, (100, 200), (40, 200))
I2 = pinholeCamera((400, 400), pinholeSize, (200, 100), (100, 40))

# Display simulated images
plotImages([I1, I2], ['Pinhole camera image 1', 'Pinhole camera image 2'])
