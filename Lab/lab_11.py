import cv2
import numpy as np
from matplotlib import pyplot as plt

# task 1

# input_image = cv2.imread("Lab/images/lab_11_2.tif", cv2.IMREAD_GRAYSCALE)
# fourier_transform = np.fft.fft2(input_image)
# fshift = np.fft.fftshift(fourier_transform)
# magnitude_spectrum = 20 * np.log(np.abs(fshift))
# rows, cols = input_image.shape
# crow, ccol = rows // 2, cols // 2
# low_pass_filter = np.zeros((rows, cols), np.uint8)
# cut_off_frequency = 30
# low_pass_filter[crow - cut_off_frequency:crow + cut_off_frequency, ccol - cut_off_frequency:ccol + cut_off_frequency] = 1
# fshift_filtered = fshift * low_pass_filter
# filtered_image = np.fft.ifftshift(fshift_filtered)
# output_image = np.fft.ifft2(filtered_image)
# output_image = np.abs(output_image)
# plt.subplot(131), plt.imshow(input_image, cmap='gray')
# plt.title('Input Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(132), plt.imshow(magnitude_spectrum, cmap='gray')
# plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
# plt.subplot(133), plt.imshow(output_image, cmap='gray')
# plt.title('Output Image'), plt.xticks([]), plt.yticks([])
# plt.show()

# task 2

# input_image = cv2.imread("Lab/images/lab_11_3.tif", cv2.IMREAD_GRAYSCALE)
# fourier_transform = np.fft.fft2(input_image)
# fshift = np.fft.fftshift(fourier_transform)
# magnitude_spectrum = 20 * np.log(np.abs(fshift))
# rows, cols = input_image.shape
# crow, ccol = rows // 2, cols // 2
# high_pass_filter = np.ones((rows, cols), np.uint8)
# cut_off_frequency = 30
# high_pass_filter[crow - cut_off_frequency:crow + cut_off_frequency, ccol - cut_off_frequency:ccol + cut_off_frequency] = 0
# fshift_filtered = fshift * high_pass_filter
# filtered_image = np.fft.ifftshift(fshift_filtered)
# gradient_image = np.fft.ifft2(filtered_image)
# gradient_image = np.abs(gradient_image)
# plt.subplot(131), plt.imshow(input_image, cmap='gray')
# plt.title('Input Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(132), plt.imshow(magnitude_spectrum, cmap='gray')
# plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
# plt.subplot(133), plt.imshow(gradient_image, cmap='gray')
# plt.title('Magnitude Gradient'), plt.xticks([]), plt.yticks([])
# plt.show()

# task 3
def bandstop_filter(shape, cutoff_frequency, width):
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    filter_mask = np.ones((rows, cols), np.uint8)
    
    for u in range(rows):
        for v in range(cols):
            D = np.sqrt((u - crow) ** 2 + (v - ccol) ** 2)
            if (cutoff_frequency - width / 2) <= D <= (cutoff_frequency + width / 2):
                filter_mask[u, v] = 0
    
    return filter_mask

input_image = cv2.imread("Lab/images/lab_11_4.tif", cv2.IMREAD_GRAYSCALE)
fourier_transform = np.fft.fft2(input_image)
fshift = np.fft.fftshift(fourier_transform)
magnitude_spectrum = 20 * np.log(np.abs(fshift))
cutoff_frequency = 25
width = 30
bandstop_filter_mask = bandstop_filter(input_image.shape, cutoff_frequency, width)
fshift_filtered = fshift * bandstop_filter_mask
filtered_image = np.fft.ifftshift(fshift_filtered)
output_image = np.fft.ifft2(filtered_image)
output_image = np.abs(output_image)

plt.subplot(131), plt.imshow(input_image, cmap='gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(132), plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.subplot(133), plt.imshow(output_image, cmap='gray')
plt.title('Filtered Image'), plt.xticks([]), plt.yticks([])
plt.show()
