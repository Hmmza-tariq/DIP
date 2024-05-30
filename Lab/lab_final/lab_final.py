import cv2
import numpy as np

image = cv2.imread('Lab\lab_final\images\lab final.tif')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blurred, 30, 150)
rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
dilated = cv2.dilate(edges, rect_kernel, iterations=1)
eroded = cv2.erode(dilated, rect_kernel, iterations=1)
contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

possible_plates = []
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = w / float(h)
    if 2 <= aspect_ratio <= 5:
        if w > 30 and h > 10:
            possible_plates.append((x, y, w, h))
possible_plates = sorted(possible_plates, key=lambda x: -x[2]*x[3])

if possible_plates:
    (x, y, w, h) = possible_plates[0]
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
cv2.imshow('image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
