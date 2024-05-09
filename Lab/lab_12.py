import skimage.io
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from scipy import fftpack

#task 1

# image1 = cv2.imread('Lab/images/lab_12_1_1.tif')
# image2 = cv2.imread('Lab/images/lab_12_1_2.tif')
# image3 = cv2.imread('Lab/images/lab_12_1_3.tif')

# image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
# image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
# image3_gray = cv2.cvtColor(image3, cv2.COLOR_BGR2GRAY)

# distances = [1, 2, 3]  
# angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]  

# my_GLCM_1 = skimage.feature.graycomatrix(image1_gray, distances, angles, levels=None, symmetric=False, normed=False)
# my_GLCM_2 = skimage.feature.graycomatrix(image2_gray, distances, angles, levels=None, symmetric=False, normed=False)
# my_GLCM_3 = skimage.feature.graycomatrix(image3_gray, distances, angles, levels=None, symmetric=False, normed=False)

# properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']

# properties1 = {prop: skimage.feature.graycoprops(my_GLCM_1, prop=prop) for prop in properties}
# properties2 = {prop: skimage.feature.graycoprops(my_GLCM_2, prop=prop) for prop in properties}
# properties3 = {prop: skimage.feature.graycoprops(my_GLCM_3, prop=prop) for prop in properties}

# print("--------GLCM Properties for Image 1--------")
# for prop, value in properties1.items():
#     print(f"\n\n{prop}: {value}")

# print("\n--------GLCM Properties for Image 2--------")
# for prop, value in properties2.items():
#     print(f"\n\n{prop}: {value}")

# print("\n--------GLCM Properties for Image 3--------")
# for prop, value in properties3.items():
#     print(f"\n\n{prop}: {value}")

# task 2

# def spectral_analysis(image):
#     fft_image = fftpack.fft2(image)
#     fft_shifted = fftpack.fftshift(fft_image)
#     magnitude_spectrum = np.abs(fft_shifted)
#     rows, cols = image.shape
#     freq_rows = np.fft.fftfreq(rows)
#     freq_cols = np.fft.fftfreq(cols)
#     freq_rows = np.fft.fftshift(freq_rows)
#     freq_cols = np.fft.fftshift(freq_cols)

#     theta = np.arctan2(freq_cols, freq_rows)
#     r = np.sqrt(freq_rows**2 + freq_cols**2)

#     S_theta = np.mean(magnitude_spectrum, axis=0)  
#     S_r = np.mean(magnitude_spectrum, axis=1)  

#     return r, theta, S_r, S_theta

# for path in ['Lab/images/lab_12_2_1.tif','Lab/images/lab_12_2_2.tif']:
#     image = cv2.imread(path)
#     gray_image1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     r, theta, S_r, S_theta = spectral_analysis(gray_image1)

#     plt.figure(figsize=(10, 5))
#     plt.subplot(1, 2, 1)
#     plt.plot(r, S_r)
#     plt.title('Feature Profile S(r)')
#     plt.xlabel('r')
#     plt.ylabel('S(r)')

#     plt.subplot(1, 2, 2)
#     plt.plot(theta, S_theta)
#     plt.title('Feature Profile S(theta)')
#     plt.xlabel('Theta (radians)')
#     plt.ylabel('S(theta)')
#     plt.show()

# task 3

def calculate_lbp_pixel(image, center_x, center_y):
    center_pixel = image[center_y, center_x]
    binary_pattern = 0
    for i, (dx, dy) in enumerate([(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]):
        x = center_x + dx
        y = center_y + dy
        if image[y, x] >= center_pixel:
            binary_pattern += 2**i
    return binary_pattern

def apply_lbp(image):
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = image.shape
    lbp_image = np.zeros_like(image, dtype=np.uint8)
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            lbp_image[y, x] = calculate_lbp_pixel(image, x, y)
    return lbp_image

image = cv2.imread('Lab/images/lab_12_3.tif')
lbp_image = apply_lbp(image)
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(lbp_image, cmap='gray')
plt.title('LBP Image')
plt.axis('off')

plt.show()


