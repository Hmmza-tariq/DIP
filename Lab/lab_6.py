import numpy as np
import cv2
import matplotlib.pyplot as plt

# Task 1
# def create_mask(mask_size, values):
#     mask = np.array(values).reshape(mask_size)
#     return mask

# def add_padding(image, padding_size, mode='zero'):
#     if mode == 'zero':
#         padded_image = np.pad(image, padding_size, mode='constant', constant_values=0)
#     elif mode == 'copy':
#         padded_image = np.pad(image, padding_size, mode='edge')
#     return padded_image

# def apply_filter(image, mask,padding_size=(1, 1)):
#     filtered_image = np.zeros(image.shape)  
#     for i in range(2, image.shape[0] - padding_size[0]):
#         for j in range(2, image.shape[1] - padding_size[1]):
#             filtered_image[i, j] = np.sum(image[i-2:i+3, j-2:j+3] * mask)
#     return filtered_image

# def normalize_image(image):
#     normalized_image = cv2.normalize(image, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
#     return normalized_image

# image = cv2.imread('Lab/images/lab5task1.tif', cv2.IMREAD_GRAYSCALE)

# mask_rows = int(input("Enter number of rows for the mask: "))
# mask_cols = int(input("Enter number of columns for the mask: "))
# mask_size = (mask_rows, mask_cols)

# mask_values = []
# print("Enter values for the mask:")
# value = float(input())
# for _ in range(mask_rows * mask_cols):
#     mask_values.append(value)

# mask = create_mask(mask_size, mask_values)

# padding_size = (mask_rows // 2, mask_cols // 2)
# padded_image_1 = add_padding(image, padding_size, mode='zero')
# filtered_image_1 = apply_filter(padded_image_1, mask, padding_size)
# normalized_image_1 = normalize_image(filtered_image_1)
# padded_image_2 = add_padding(image, padding_size, mode='copy')
# filtered_image_2 = apply_filter(padded_image_2, mask, padding_size)
# normalized_image_2 = normalize_image(filtered_image_2)

# plt.figure(figsize=(10, 6))

# plt.subplot(2, 2, 1)
# plt.imshow(image, cmap='gray')
# plt.title('Original Image')
# plt.axis('off')

# plt.subplot(2, 2, 3)
# plt.imshow(normalized_image_1, cmap='gray')
# plt.title('Processed Image: zero')
# plt.axis('off')

# plt.subplot(2, 2, 4)
# plt.imshow(normalized_image_2, cmap='gray')
# plt.title('Processed Image: copy')
# plt.axis('off')

# plt.tight_layout()
# plt.show()

# Task 2

image = cv2.imread('Lab/images/lab_5.tif', cv2.IMREAD_GRAYSCALE)

def min_filter(image, kernel_size):
    padding = kernel_size // 2
    padded_image = np.pad(image, padding, mode='edge')
    filtered_image = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            filtered_image[i, j] = np.min(padded_image[i:i+kernel_size, j:j+kernel_size])
    return filtered_image

def max_filter(image, kernel_size):
    padding = kernel_size // 2
    padded_image = np.pad(image, padding, mode='edge')
    filtered_image = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            filtered_image[i, j] = np.max(padded_image[i:i+kernel_size, j:j+kernel_size])
    return filtered_image

neighborhood_sizes = [3, 5]
filtered_images = {}
for size in neighborhood_sizes:
    min_filtered = min_filter(image, size)
    max_filtered = max_filter(image, size)
    filtered_images[f'Min Filter {size}x{size}'] = min_filtered
    filtered_images[f'Max Filter {size}x{size}'] = max_filtered

plt.figure(figsize=(12, 8))
plt.subplot(2, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')

for i, (title, filtered_image) in enumerate(filtered_images.items(), start=2):
    plt.subplot(2, 3, i)
    plt.imshow(filtered_image, cmap='gray')
    plt.title(title)

plt.tight_layout()
plt.show()

# Task 3

# image = cv2.imread('Lab/images/lab5task3.tif', cv2.IMREAD_GRAYSCALE)

# filtered_image = cv2.medianBlur(image, 3)  
# plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1)
# plt.imshow(image, cmap='gray')
# plt.title('Noisy Image')
# plt.subplot(1, 2, 2)
# plt.imshow(filtered_image, cmap='gray')
# plt.title('Filtered Image (Median Filter)')
# plt.show()