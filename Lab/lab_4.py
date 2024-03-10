import cv2
import numpy as np
import matplotlib.pyplot as plt


# Task 1
# image_1 = cv2.imread('lab_4_1.tif')
# image_2 = cv2.imread('lab_4_2.tif')
#
# # s = (L – 1) – r
# negative_image_1 = 255 - image_1
# negative_image_2 = 255 - image_2
#
# # c=255/log(1+max of image pixel), s=c*log(img+1)
# c = 255 / np.log(1 + np.max(image_1))
# log_image_1 = c * (np.log(image_1 + 1))
#
# c = 255 / np.log(1 + np.max(image_2))
# log_image_2 = c * (np.log(image_2 + 1))
#
# negative_image_1 = negative_image_1.astype(np.uint8)
# log_image_1 = log_image_1.astype(np.uint8)
# negative_image_2 = negative_image_2.astype(np.uint8)
# log_image_2 = log_image_2.astype(np.uint8)
#
# grid_1 = np.hstack((image_1, negative_image_1, log_image_1))
# cv2.imshow('Original Image, Negative Transformation, Log Transformation', grid_1)
# cv2.waitKey(0)
#
# grid_2 = np.hstack((image_2, negative_image_2, log_image_2))
# cv2.imshow('Original Image, Negative Transformation, Log Transformation', grid_2)
#
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Task 2

# def part_a(image, mean):
#     for i in range(image.shape[0]):
#         for j in range(image.shape[1]):
#             if image[i, j] > mean:
#                 image[i, j] = 255
#             else:
#                 image[i, j] = 0
#     return image
#
#
# def part_b(image, mean):
#     for i in range(image.shape[0]):
#         for j in range(image.shape[1]):
#             if image[i, j] < mean:
#                 image[i, j] = 255
#             else:
#                 image[i, j] = 0
#     return image
#
#
# def part_c(image, mean):
#     for i in range(image.shape[0]):
#         for j in range(image.shape[1]):
#             if image[i, j] > mean+20 or image[i, j] < mean-20:
#                 image[i, j] = 255
#             else:
#                 image[i, j] = 0
#     return image
#
#
# image_1_1 = cv2.imread('lab_4_1.tif', cv2.IMREAD_GRAYSCALE)
# image_1_2 = cv2.imread('lab_4_1.tif', cv2.IMREAD_GRAYSCALE)
# image_1_3 = cv2.imread('lab_4_1.tif', cv2.IMREAD_GRAYSCALE)
#
# image_2_1 = cv2.imread('lab_4_2.tif', cv2.IMREAD_GRAYSCALE)
# image_2_2 = cv2.imread('lab_4_2.tif', cv2.IMREAD_GRAYSCALE)
# image_2_3 = cv2.imread('lab_4_2.tif', cv2.IMREAD_GRAYSCALE)
#
# mean_image_1 = np.mean(image_1_1)
# mean_image_2 = np.mean(image_2_1)
#
# part_a(image_1_1, mean_image_1)
# part_a(image_2_1, mean_image_2)
#
# part_b(image_1_2, mean_image_1)
# part_b(image_2_2, mean_image_2)
#
# part_c(image_1_3, mean_image_1)
# part_c(image_2_3, mean_image_2)
#
# grid_1 = np.hstack((image_1_1, image_1_2, image_1_3))
# cv2.imshow('Part A, Part B, Part C', grid_1)
#
# cv2.waitKey(0)
#
# grid_2 = np.hstack((image_2_1, image_2_2, image_2_3))
# cv2.imshow('Part A, Part B, Part C', grid_2)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Task 3
# def power_transformation(image):
#     gammas = [0.2, 0.5, 1.2, 1.8]
#     images = []
#     for gamma in gammas:
#         power_law_image = np.power(image / 255.0, gamma) * 255
#         power_law_image = np.uint8(power_law_image)
#         images.append(power_law_image)
#     return images
#
#
# image_1 = cv2.imread('lab_4_1.tif', cv2.IMREAD_GRAYSCALE)
# image_2 = cv2.imread('lab_4_2.tif', cv2.IMREAD_GRAYSCALE)
#
#
# images_1 = power_transformation(image_1)
# images_2 = power_transformation(image_2)
#
# grid_1 = np.hstack(images_1)
# cv2.imshow('Power Law Transformations', grid_1)
# cv2.waitKey(0)
#
# grid_2 = np.hstack(images_2)
# cv2.imshow('Power Law Transformations', grid_2)
# cv2.waitKey(0)
#
#
# cv2.destroyAllWindows()

# Task 4

# image_1 = cv2.imread('lab_4_1.tif', cv2.IMREAD_GRAYSCALE)
# image_2 = cv2.imread('lab_4_2.tif', cv2.IMREAD_GRAYSCALE)
#
#
# lower_limit = 100
# upper_limit = 200
# new_value = 210
#
# image_1 = np.where((image_1 >= lower_limit) & (image_1 <= upper_limit), new_value, image_1)
# image_2 = np.where((image_2 >= lower_limit) & (image_2 <= upper_limit), new_value, image_2)
#
# cv2.imshow('Gray Level Slicing', image_1)
# cv2.waitKey(0)
# cv2.imshow('Gray Level Slicing', image_2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Task 5

# Load the image
# image = cv2.imread('lab_4_2.tif', cv2.IMREAD_GRAYSCALE)
# histogram = np.zeros(256)
#
# for i in range(image.shape[0]):
#     for j in range(image.shape[1]):
#         histogram[image[i, j]] += 1
#
# plt.bar(range(256), histogram)
# plt.title('Histogram')
# plt.xlabel('Pixel Value')
# plt.ylabel('Frequency')
# plt.show()

# Task 6
#
# def calculate_mean(pixel_values):
#     return np.mean(pixel_values)
#
#
# def calculate_variance(pixel_values):
#     return np.var(pixel_values)
#
#
# def enhance_image(image, k0, k1, k2, E):
#     global_mean = np.mean(image)
#     global_std = np.std(image)
#
#     enhanced_image = np.copy(image)
#
#     rows, cols = image.shape[:2]
#
#     for i in range(rows):
#         for j in range(cols):
#             start_row = max(0, i - 1)
#             end_row = min(rows, i + 2)
#             start_col = max(0, j - 1)
#             end_col = min(cols, j + 2)
#
#             neighborhood = image[start_row:end_row, start_col:end_col]
#
#             mean_val = calculate_mean(neighborhood)
#             var_val = calculate_variance(neighborhood)
#
#             if (mean_val > k0 * global_mean) and (var_val < k2 * global_std) and (
#                     k1 * global_std < var_val < k2 * global_std):
#                 enhanced_image[i, j] *= E
#
#     return enhanced_image
#
#
# image = cv2.imread("lab_4_2.tif", cv2.IMREAD_GRAYSCALE)
# k0 = 0.4
# k1 = 0.02
# k2 = 0.4
# E = 4
# enhanced_image = enhance_image(image, k0, k1, k2, E)
#
# output_image = np.hstack((image, enhanced_image))
# cv2.imshow("Original and Enhanced Image", output_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
