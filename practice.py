import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the images
mask_image = cv2.imread('Assignment-1\Test\Tissue\RA23-01883-B1-2-PAS.[17408x1536].jpg', cv2.IMREAD_GRAYSCALE)
live_image = cv2.imread('Assignment-1\Test\Tissue\RA23-01882-A1-1-PAS.[10240x2048].jpg', cv2.IMREAD_GRAYSCALE)

# Display the images
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Mask Image')
plt.imshow(mask_image, cmap='gray')
plt.subplot(1, 2, 2)
plt.title('Live Image')
plt.imshow(live_image, cmap='gray')
plt.show()
# Subtract the mask image from the live image
subtracted_image = cv2.subtract(live_image, mask_image)

# Display the subtracted image
plt.figure(figsize=(5, 5))
plt.title('Subtracted Image')
plt.imshow(subtracted_image, cmap='gray')
plt.show()
# Apply a binary threshold to enhance the visibility of blood vessels
_, thresholded_image = cv2.threshold(subtracted_image, 50, 255, cv2.THRESH_BINARY)

# Apply morphological operations to remove noise and enhance structures
kernel = np.ones((3, 3), np.uint8)
morph_image = cv2.morphologyEx(thresholded_image, cv2.MORPH_OPEN, kernel, iterations=2)
morph_image = cv2.morphologyEx(morph_image, cv2.MORPH_CLOSE, kernel, iterations=2)

# Display the post-processed image
plt.figure(figsize=(5, 5))
plt.title('Post-Processed Image')
plt.imshow(morph_image, cmap='gray')
plt.show()
# Save the final image
cv2.imwrite('final_result.jpg', morph_image)
