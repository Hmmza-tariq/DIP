import cv2
import numpy as np

# Load imageAssignment-2\Fundus image\23_training.tif
image_path = "Assignment-2/Fundus image/02_test.tif"  # Replace with actual path to the image
image = cv2.imread(image_path)

# Resize image to 512x512
resized_image = cv2.resize(image, (512, 512))

# Separate RGB channels
blue_channel, green_channel, red_channel = cv2.split(resized_image)

# Select green channel
green_channel_selected = green_channel

# Apply Gaussian filter
green_blurred = cv2.GaussianBlur(green_channel_selected, (49, 49), 0)

# Apply opening filter (erosion followed by dilation)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
opened_image = cv2.morphologyEx(green_blurred, cv2.MORPH_OPEN, kernel)

# Set the size of region of interest to 110x110 and extract that from the original image
roi_size = 110
center = (resized_image.shape[1] // 2, resized_image.shape[0] // 2)
roi_x = center[0] - roi_size // 2
roi_y = center[1] - roi_size // 2
roi = resized_image[roi_y:roi_y + roi_size, roi_x:roi_x + roi_size]

# Convert region of interest to grayscale
roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

# Apply Gaussian filter to blur it
roi_blurred = cv2.GaussianBlur(roi_gray, (5, 5), 0)

# Blend blurred image with the original region of interest to sharpen it
sharpened_roi = cv2.addWeighted(roi_gray, 1.5, roi_blurred, -0.5, 0)

# Apply closing operation (dilation followed by erosion)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
closed_roi = cv2.morphologyEx(sharpened_roi, cv2.MORPH_CLOSE, kernel)

# Equalize the resulting image
equalized_roi = cv2.equalizeHist(closed_roi)

# Use Canny edge detection on this equalized image
edges = cv2.Canny(equalized_roi, 30, 150)

# Dilate the edges
dilated_edges = cv2.dilate(edges, None)

# Apply Hough circular transform
circles = cv2.HoughCircles(dilated_edges, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
                            param1=50, param2=30, minRadius=15, maxRadius=40)

# Optic disc is located and a circle is drawn around it and its centre in marked with a red dot
if circles is not None:
    circles = np.round(circles[0, :]).astype("int")
    for (x, y, r) in circles:
        cv2.circle(resized_image, (x + roi_x, y + roi_y), r, (0, 255, 0), 4)
        cv2.circle(resized_image, (x + roi_x, y + roi_y), 3, (0, 0, 255), -1)

# Show the result
cv2.imshow("Result", resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
