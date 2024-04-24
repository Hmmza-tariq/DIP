import cv2
import numpy as np

# Read the input image
image = cv2.imread('Assignment-2/Fundus-image/29_training.tif', cv2.IMREAD_COLOR)
resized_image = cv2.resize(image, (512, 512))

# Convert the image to grayscale
gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise
blurred = cv2.GaussianBlur(gray, (3, 3), 0)

# Apply Canny edge detection to detect vessel edges
edges = cv2.Canny(gray, 30, 150)
    # Display the result
cv2.imshow("Vessel", edges)
cv2.waitKey(0)
cv2.destroyAllWindows()

# import cv2
# import numpy as np
# import math

# def canny(img, sigma):
#  # find mean of the image
#  avg = np.mean(img)
#  # find the lower limit utilizing sigma and mean
#  lower = int(max(0, (1.0 - sigma) * avg))
#  # find the upper limit utilizing sigma and mean
#  upper = int(min(255, (1.0 + sigma) * avg))
#  # return the image applied with canny detector
#  return cv2.Canny(img, lower, upper)

# def circle_perimeter(radius):
#     """Calculate the perimeter (circumference) of a circle given its radius."""
#     return 2 * math.pi * radius

# # Load an example retinal image
# image = cv2.imread('Assignment-2/Fundus-image/29_training.tif')

# # 1. Resize image to 512x512
# resized_image = cv2.resize(image, (512, 512))

# # 2. Separate RGB channels
# b, g, r = cv2.split(resized_image)

# # 3. Select the green channel
# green_channel = g

# # 4. Apply Gaussian filter size 49x49
# green_channel_blurred = cv2.GaussianBlur(green_channel, (45, 45), 0)

# # 5. Apply opening filter (erosion followed by dilation) with circular kernel
# kernel_opening = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
# green_channel_opening = cv2.morphologyEx(green_channel_blurred, cv2.MORPH_OPEN, kernel_opening)

# # 6. Set the size of region of interest to 110x110 and extract that from the original image
# (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(green_channel_opening)
#  # now create a window of our region of interest around that pixel
#  # window size of 110 provides better results
# window = 110
# x0 = abs(int(maxLoc[0]) - window)
# y0 = abs(int(maxLoc[1]) - window)
# x1 = abs(int(maxLoc[0]) + window)
# y1 = abs(int(maxLoc[1]) + window)
#  # finally, return the 110x110 window
# roi=resized_image[y0:y1, x0:x1]

# # 7. Apply Gaussian filter to blur it
# roi_blurred = cv2.GaussianBlur(roi, (5, 5), 0)

# # 8. Blurred image is blended with the original image from step 7 to sharpen it
# sharpened_image = cv2.addWeighted(roi, 1.5, roi_blurred, -0.5, 0)

# # 9. Apply closing operation (dilation followed by erosion)
# kernel_closing = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (16, 16))
# sharpened_image_closing = cv2.morphologyEx(sharpened_image, cv2.MORPH_CLOSE, kernel_closing)
# # Convert the image to grayscale if it's not already in grayscale
# if len(sharpened_image_closing.shape) > 2:
#     sharpened_image_closing = cv2.cvtColor(sharpened_image_closing, cv2.COLOR_BGR2GRAY)
# # 10. Equalize the resulting image
# equalized_image = cv2.equalizeHist(sharpened_image_closing)

# # 11. Use Canny edge detection on this equalized image
# edges = canny(equalized_image, 0.22)

# # 12. Output is dilated
# dilated_edges = cv2.dilate(edges, None, iterations=1)
# # kernel_closing = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (16, 16))
# # dil_closing = cv2.morphologyEx(sharpened_image, cv2.MORPH_CLOSE, kernel_closing)
# # 13. Apply Hough circular transform
# circles = cv2.HoughCircles(dilated_edges, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=30, param2=15, minRadius=55, maxRadius=80)
# if circles is not None:
#     # Convert the coordinates and radii to integers
#     circles = np.round(circles[0, :]).astype("int")
#     print(circles)
#     # Iterate over all detected circles
#     for (cx, cy, radius) in circles:
#         # Draw the circle outline
#         cv2.circle(resized_image, (cx, cy), radius, (0, 255, 0), 2)
        
#         # Draw the center of the circle
#         cv2.circle(resized_image, (cx, cy), 2, (0, 0, 0), 5)
# else:
#     print("No circles detected.")



# # Display the results
# cv2.imshow("green", green_channel)
# cv2.imshow("blurred", green_channel_blurred)
# cv2.imshow("opened", green_channel_opening)
# cv2.imshow("roi", roi)
# cv2.imshow("roi_blurred", roi_blurred)
# cv2.imshow("sharpened", sharpened_image)
# cv2.imshow("sharp_close", sharpened_image_closing)
# cv2.imshow('equalized',equalized_image)
# cv2.imshow("edges", edges)
# cv2.imshow("dil-edges", dilated_edges)
# cv2.imshow("marked", resized_image)

# # cv2.imshow("Optic Disc Detection", resized_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()