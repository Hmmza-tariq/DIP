import math
import cv2
import numpy as np

# task 1
image = cv2.imread('Lab/images/lab_7_1.jpg', cv2.IMREAD_GRAYSCALE)

sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)

magnitude = np.sqrt(sobelx**2 + sobely**2)

magnitude /= np.max(magnitude)


top_30_percent = np.percentile(magnitude, 70)

magnitude_thresholded = np.where(magnitude >= top_30_percent, magnitude, 0)

phase = np.arctan2(sobely, sobelx)
phase_degrees = np.degrees(phase)

phase_45 = np.where(np.logical_and(phase_degrees >= 40, phase_degrees <= 50), magnitude, 0)
phase_90 = np.where(np.logical_or(np.logical_and(phase_degrees >= 80, phase_degrees <= 100),
                                  np.logical_and(phase_degrees >= -10, phase_degrees <= 10)), magnitude, 0)


cv2.imshow('Magnitude', magnitude_thresholded)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow('45 Degrees', phase_45)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow('90 Degrees', phase_90)
cv2.waitKey(0)
cv2.destroyAllWindows()

# task 2
# image = cv2.imread('Lab/images/lab_7_2.png', cv2.IMREAD_GRAYSCALE)

# global_mean = np.mean(image)
# global_median = np.median(image)

# _, thresh_mean = cv2.threshold(image, global_mean, 255, cv2.THRESH_BINARY)
# _, thresh_median = cv2.threshold(image, global_median, 255, cv2.THRESH_BINARY)

# cv2.imshow('Original Image', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# cv2.imshow('Thresholded Image (Global Mean)', thresh_mean)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# cv2.imshow('Thresholded Image (Global Median)', thresh_median)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# task 3
# image = cv2.imread('Lab/images/lab_7_2.png', cv2.IMREAD_GRAYSCALE)

# height, width = image.shape[:2]

# block_size = 3

# output = np.zeros_like(image)

# for i in range(0, height, block_size):
#     for j in range(0, width, block_size):
#         block = image[i:i+block_size, j:j+block_size]
#         block_mean = np.mean(block)
#         output[i:i+block_size, j:j+block_size] = np.where(block > block_mean + 3.5, 255, 0)

# cv2.imshow('Original Image', image)
# cv2.imshow('Locally Thresholded Image (Mean)', output)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# task 4

image = cv2.imread('Lab/images/lab_7_4.bmp')
pixels = image.reshape((-1, 3))
pixels = np.float32(pixels)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
k = 2

centers = np.random.randint(0, 256, (k, 3)).astype(np.float32)

for _ in range(10):
    distances = np.sqrt(np.sum(np.square(pixels[:, None] - centers), axis=-1))
    labels = np.argmin(distances, axis=1)

    for i in range(k):
        centers[i] = np.mean(pixels[labels == i], axis=0)

centers = np.uint8(centers)
segmented_image = centers[labels.flatten()]
segmented_image = segmented_image.reshape(image.shape)

cv2.imshow('Original Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow('Segmented Image', segmented_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

