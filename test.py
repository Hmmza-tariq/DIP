import numpy as np
import cv2


def my_dilate(image, kernel):
    # Get the dimensions of the image and kernel
    img_height, img_width = image.shape
    kernel_height, kernel_width = kernel.shape

    # Calculate the padding needed for the image to apply the kernel
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2

    # Create a padded version of the image
    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant')

    # Create an empty output image
    output_image = np.zeros_like(image)

    # Perform dilation
    for i in range(img_height):
        for j in range(img_width):
            # Extract the region of interest (ROI) from the padded image
            roi = padded_image[i:i + kernel_height, j:j + kernel_width]

            # Check if any pixel under the kernel is white (255)
            if np.any(roi == 255):  
                output_image[i, j] = 255  # Set the pixel to white (255)
            else:
                output_image[i, j] = 0  # Set the pixel to black (0)

    return output_image

def my_erode(image, kernel):
    # Get the dimensions of the image and kernel
    img_height, img_width = image.shape
    kernel_height, kernel_width = kernel.shape

    # Calculate the padding needed for the image to apply the kernel
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2

    # Create a padded version of the image
    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant')

    # Create an empty output image
    output_image = np.zeros_like(image)

    # Perform erosion
    for i in range(img_height):
        for j in range(img_width):
            # Extract the region of interest (ROI) from the padded image
            roi = padded_image[i:i + kernel_height, j:j + kernel_width]

            # Check if all pixels under the kernel are non-zero
            if np.all(roi == 255):  # Check if all pixels are white (255)
                output_image[i, j] = 255  # Set the pixel to white (255)
            else:
                output_image[i, j] = 0  # Set the pixel to black (0)

    return output_image

kernel_size = 19
structuring_element = np.ones((kernel_size, kernel_size), dtype=np.uint8)

image = cv2.imread('Lab/images/lab_9_4.jpg', 0)  # Load as grayscale

eroded_image = my_erode(image, structuring_element)

cv2.imshow('Original Image', image)
cv2.imshow('Eroded Image', eroded_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# # Task 2

kernel_size = 2

structuring_element = np.ones((kernel_size, kernel_size), dtype=np.uint8)

image = cv2.imread('fingerprint.tif', 0)  # Load as grayscale

eroded_image = my_erode(image, structuring_element)
dilated_image = my_dilate(eroded_image, structuring_element)


cv2.imshow('Original Image', image)
cv2.imshow('Fingerprint Clear Image', dilated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# # Task 3

kernel_size = 3

structuring_element = np.ones((kernel_size, kernel_size), dtype=np.uint8)

image = cv2.imread('headCT.tif', 0)  # Load as grayscale

dilated_image = my_dilate(image, structuring_element)

eroded_image = my_erode(image, structuring_element)

final_image = dilated_image - eroded_image


cv2.imshow('Original Image', image)
cv2.imshow('Fingerprint Clear Image', final_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Task 4

kernel_size = 9

structuring_element = np.ones((kernel_size, kernel_size), dtype=np.uint8)

image = cv2.imread('rice.tif', 0)  # Load as grayscale

eroded_image = my_erode(image, structuring_element)
final_image = my_dilate(eroded_image, structuring_element)

final_image_2 = image - final_image


cv2.imshow('Original Image', image)
cv2.imshow('Fingerprint Clear Image', final_image_2)
cv2.waitKey(0)
cv2.destroyAllWindows()