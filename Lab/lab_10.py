import cv2
import numpy as np

# Load RGB image
rgb_image = cv2.imread('Lab/images/1a.tif')

# Convert to float32 and normalize to range [0, 1]
rgb_image = rgb_image.astype(np.float32) / 255.0

# Split the RGB image into individual channels
B, G, R = cv2.split(rgb_image)

# Calculate Cmax and Cmin
Cmax = np.maximum(np.maximum(R, G), B)
Cmin = np.minimum(np.minimum(R, G), B)
delta = Cmax - Cmin

# Calculate Hue
H = np.zeros_like(Cmax)
mask = delta != 0
H[mask] = np.where(Cmax == R, 60 * ((G - B) / delta + 0), 0)
H[mask] += np.where(Cmax == G, 60 * ((B - R) / delta + 2), 0)
H[mask] += np.where(Cmax == B, 60 * ((R - G) / delta + 4), 0)
H[mask] %= 360

# Calculate Saturation
S = np.zeros_like(Cmax)
mask = Cmax != 0
S[mask] = delta[mask] / Cmax[mask]

# Calculate Value
V = Cmax

# Display each channel
cv2.imshow("Hue", H.astype(np.uint8))
cv2.imshow("Saturation", (S * 255).astype(np.uint8))
cv2.imshow("Value", (V * 255).astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()
