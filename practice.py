import numpy as np
from scipy.signal import convolve2d

# Given 2x2 image
original_image = np.array([[2, 3],
                           [4, 5]])

# Upsample the image by inserting zeros
upsampled_image = np.zeros((4, 4))
upsampled_image[::2, ::2] = original_image

# Define the spatial domain low-pass filter
low_pass_filter = np.array([[0.25, 0.5, 0.25],
                             [0.5, 1, 0.5],
                             [0.25, 0.5, 0.25]])

# Convolve the upsampled image with the low-pass filter
interpolated_image = convolve2d(upsampled_image, low_pass_filter, mode='same', boundary='fill', fillvalue=0)

print("Interpolated Image:")
print(interpolated_image)
