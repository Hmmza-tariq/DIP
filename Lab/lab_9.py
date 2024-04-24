import cv2
import numpy as np

def custom_erosion(image, kernel):
    _, image = cv2.threshold(image, 120, 255, cv2.THRESH_BINARY)
    padding = kernel.shape[0] // 2
    image = cv2.copyMakeBorder(image, padding, padding, padding, padding, cv2.BORDER_CONSTANT, None, value=0)
    eroded_image = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    for i in range(padding, image.shape[0]-padding):
        for j in range(padding, image.shape[1] - padding):
            if np.count_nonzero(np.multiply(image[i-padding:i+padding+1, j-padding:j+padding+1], kernel)) == np.count_nonzero(kernel):
                eroded_image[i, j] = 255

    return eroded_image[padding:image.shape[0]-padding, padding:image.shape[1]-padding]

def custom_dilation(image, kernel):
    _, image = cv2.threshold(image, 120, 255, cv2.THRESH_BINARY)
    padding = kernel.shape[0] // 2
    image = cv2.copyMakeBorder(image, padding, padding, padding, padding, cv2.BORDER_CONSTANT, None, value=0)
    dilated_image = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    for i in range(padding, image.shape[0]-padding):
        for j in range(padding, image.shape[1] - padding):
            if np.count_nonzero(np.multiply(image[i-padding:i+padding+1, j-padding:j+padding+1], kernel)) > 0:
                dilated_image[i, j] = 255

    return dilated_image[padding:image.shape[0]-padding, padding:image.shape[1]-padding]




def count_balls(image):
    _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_image, connectivity=8)
    num_balls = num_labels - 1

    return num_balls


# task 1
kernel = np.ones((19, 19)) * 255
image = cv2.imread('Lab/images/lab_9_1.png', cv2.IMREAD_GRAYSCALE)
eroded_image = custom_erosion(image, kernel)
num_balls = count_balls(eroded_image)
cv2.imshow('Original Image', image)
cv2.imshow('Eroded Image', eroded_image)
print("Total number of balls after erosion:", num_balls)
cv2.waitKey(0)
cv2.destroyAllWindows()

# task 2
kernel = np.ones((3, 3)) * 255
image = cv2.imread('Lab/images/lab_9_2.jpg', cv2.IMREAD_GRAYSCALE)
eroded_image = custom_erosion(image, kernel)
dilated_image = custom_dilation(eroded_image, kernel)

cv2.imshow('Original Image', image)
cv2.imshow('After noise removal', eroded_image)
cv2.imshow('After filling gaps', dilated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# task 3
image = cv2.imread('Lab/images/lab_9_3.jpg', cv2.IMREAD_GRAYSCALE)
eroded_image = custom_erosion(image, kernel)
dilated_image = custom_dilation(image, kernel)
morphological_gradient = dilated_image - eroded_image
cv2.imshow('Original Image', image)
cv2.imshow('Eroded Image', eroded_image)
cv2.imshow('Dilated Image', dilated_image)
cv2.imshow('Morphological Gradient', morphological_gradient)
cv2.waitKey(0)
cv2.destroyAllWindows()

# task 4
kernel = np.ones((25, 25)) * 255
image = cv2.imread('Lab/images/lab_9_4.jpg', 0  )
opening = custom_dilation(custom_erosion(image, kernel), kernel)
topHat = image- opening
cv2.imshow('Original Image', image)
cv2.imshow('Top-hat Transformation', topHat)
cv2.waitKey(0)
cv2.destroyAllWindows()
