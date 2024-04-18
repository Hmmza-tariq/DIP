import cv2
import numpy as np

def custom_erosion(image, kernel_size):
    height, width = image.shape
    kernel_half = kernel_size // 2
    eroded_image = np.zeros_like(image)
    
    for i in range(kernel_half, height - kernel_half):
        for j in range(kernel_half, width - kernel_half):
            min_value = 255
            for k in range(-kernel_half, kernel_half + 1):
                for l in range(-kernel_half, kernel_half + 1):
                    pixel_value = image[i + k, j + l]
                    if pixel_value < min_value:
                        min_value = pixel_value
            eroded_image[i, j] = min_value
    
    return eroded_image

def custom_dilation(image, kernel_size):
    height, width = image.shape
    kernel_half = kernel_size // 2
    dilated_image = np.zeros_like(image)
    
    for i in range(kernel_half, height - kernel_half):
        for j in range(kernel_half, width - kernel_half):
            max_value = 0
            for k in range(-kernel_half, kernel_half + 1):
                for l in range(-kernel_half, kernel_half + 1):
                    pixel_value = image[i + k, j + l]
                    if pixel_value > max_value:
                        max_value = pixel_value
            dilated_image[i, j] = max_value
    
    return dilated_image


def count_balls(image):
    _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_image, connectivity=8)
    num_balls = num_labels - 1

    return num_balls

# task 1
# image = cv2.imread('Lab/images/lab_9_1.png', cv2.IMREAD_GRAYSCALE)
# eroded_image = custom_erosion(image, kernel_size=19)
# num_balls = count_balls(eroded_image)
# cv2.imshow('Original Image', image)
# cv2.imshow('Eroded Image', eroded_image)
# print("Total number of balls after erosion:", num_balls)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# task 2
# image = cv2.imread('Lab/images/lab_9_2.jpg', cv2.IMREAD_GRAYSCALE)
# eroded_image = custom_erosion(image, kernel_size=3)
# dilated_image = custom_dilation(eroded_image, kernel_size=3)

# cv2.imshow('Original Image', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# cv2.imshow('After noise removal', eroded_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# cv2.imshow('After filling gaps', dilated_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# task 3
# image = cv2.imread('Lab/images/lab_9_3.jpg', cv2.IMREAD_GRAYSCALE)
# eroded_image = custom_erosion(image, kernel_size=3)
# dilated_image = custom_dilation(image, kernel_size=3)
# morphological_gradient = dilated_image - eroded_image
# cv2.imshow('Original Image', image)
# cv2.imshow('Eroded Image', eroded_image)
# cv2.imshow('Dilated Image', dilated_image)
# cv2.imshow('Morphological Gradient', morphological_gradient)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# task 4
image = cv2.imread('Lab/images/lab_9_4.jpg', cv2.IMREAD_GRAYSCALE)
opening = custom_dilation(custom_erosion(image, kernel_size=19), kernel_size=19)
topHat = cv2.subtract(image, opening)
cv2.imshow('Original Image', image)
cv2.imshow('Top-hat Transformation', topHat)
cv2.waitKey(0)
cv2.destroyAllWindows()
