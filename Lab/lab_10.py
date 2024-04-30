import cv2
import numpy as np

# task 1
# def convert(image):
#     B, G, R = cv2.split(image)
#     Cmax = np.maximum(np.maximum(R, G), B)
#     Cmin = np.minimum(np.minimum(R, G), B)
#     delta = Cmax - Cmin
#     H = np.zeros_like(Cmax, dtype=np.float32)
#     mask = delta != 0

#     H[mask] = np.where(Cmax[mask] == R[mask], (60 * ((G[mask] - B[mask]) / delta[mask]) % 360), H[mask])
#     H[mask] += np.where(Cmax[mask] == G[mask], (60 * ((B[mask] - R[mask]) / delta[mask]) + 120), 0)
#     H[mask] += np.where(Cmax[mask] == B[mask], (60 * ((R[mask] - G[mask]) / delta[mask]) + 240), 0)

#     S = np.zeros_like(Cmax, dtype=np.float32)
#     S[mask] = delta[mask] / Cmax[mask]

#     V = Cmax

#     return H, S, V

# str_image = cv2.imread('Lab/images/STR.tif', cv2.IMREAD_COLOR)
# box_image = cv2.imread('Lab/images/BOX.jpg', cv2.IMREAD_COLOR)

# H, S, V = convert(str_image)

# cv2.imshow("Hue", (H / 2).astype(np.uint8))  
# cv2.imshow("Saturation", (S * 255).astype(np.uint8))
# cv2.imshow("Value", (V * 255).astype(np.uint8))
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# H, S, V = convert(box_image)

# cv2.imshow("Hue", (H / 2).astype(np.uint8))
# cv2.imshow("Saturation", (S * 255).astype(np.uint8))
# cv2.imshow("Value", (V * 255).astype(np.uint8))
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# task 2
# rgb_image = cv2.imread('Lab/images/lab10_2.png', cv2.IMREAD_COLOR)
# B, G, R = cv2.split(rgb_image)

# smoothed_R = cv2.GaussianBlur(R, (5, 5), 0)
# smoothed_G = cv2.GaussianBlur(G, (5, 5), 0)
# smoothed_B = cv2.GaussianBlur(B, (5, 5), 0)

# smoothed_image = cv2.merge([smoothed_B, smoothed_G, smoothed_R])

# cv2.imshow("Original Image", rgb_image)
# cv2.imshow("Smoothed Image (Separate Channels)", smoothed_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# task 3
# image = cv2.imread('Lab/images/1a.tif')

# R = image[:, :, 2]
# G = image[:, :, 1]
# B = image[:, :, 0]

# sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
# sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

# R_x = cv2.filter2D(R, -1, sobel_x)
# R_y = cv2.filter2D(R, -1, sobel_y)
# R_gradient = np.sqrt(R_x ** 2 + R_y ** 2)
# R_gradient = np.uint8(R_gradient / np.max(R_gradient) * 255)

# G_x = cv2.filter2D(G, -1, sobel_x)
# G_y = cv2.filter2D(G, -1, sobel_y)
# G_gradient = np.sqrt(G_x ** 2 + G_y ** 2)
# G_gradient = np.uint8(G_gradient / np.max(G_gradient) * 255)

# B_x = cv2.filter2D(B, -1, sobel_x)
# B_y = cv2.filter2D(B, -1, sobel_y)
# B_gradient = np.sqrt(B_x ** 2 + B_y ** 2)
# B_gradient = np.uint8(B_gradient / np.max(B_gradient) * 255)

# final_image = np.zeros_like(image)
# final_image[:, :, 2] = R_gradient
# final_image[:, :, 1] = G_gradient
# final_image[:, :, 0] = B_gradient

# cv2.imshow("Original Image", image)
# cv2.imshow("Sobel Filtered Image", final_image)
# cv2.waitKey(0)

# task 4

# Load the image
image = cv2.imread("Lab/images/1b.tif", cv2.IMREAD_GRAYSCALE)

fft_result = np.fft.fft2(image)
fft_shifted = np.fft.fftshift(fft_result)
magnitude_spectrum = 20 * np.log(np.abs(fft_shifted))

cv2.imshow("Original Image", image)
cv2.imshow("FFT", magnitude_spectrum.astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()

