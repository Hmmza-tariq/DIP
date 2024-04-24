import os
import cv2
import numpy as np

def read_images_from_folders(folder_path):
    images = []
    names = []
    files = os.listdir(folder_path)    
    files.sort()
    for file in files:       
        path = os.path.join(folder_path, file)
        names.append(file)
        path = path.replace("\\", "/")
        images.append(path)
    return images,names

def resize_image(image, new_size=(512, 512)):
    resized_image = cv2.resize(image, new_size)
    return resized_image
    
def find_brightest_spot(image):
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(grayscale_image)
    brightest_spot = max_loc
    return brightest_spot
 
def find_brightest_spots(image, threshold=200):
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Threshold the grayscale image to highlight bright spots
    _, thresholded = cv2.threshold(grayscale_image, threshold, 255, cv2.THRESH_BINARY)
    
    # Find contours of bright spots
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Get the centroids of each contour (bright spot)
    brightest_spots = []
    for contour in contours:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            centroid_x = int(M["m10"] / M["m00"])
            centroid_y = int(M["m01"] / M["m00"])
            brightest_spots.append((centroid_x, centroid_y))
    
    return brightest_spots

def extract_vessels(image):
    kernel = np.ones((3, 3), np.uint8)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    eroded_image = cv2.erode(gray_image, kernel, iterations=1)
    dilated_image = cv2.dilate(eroded_image, kernel, iterations=1)
    vessels = dilated_image - eroded_image
    _, binary_vessels = cv2.threshold(vessels, 8, 255, cv2.THRESH_BINARY)
    
    return binary_vessels

def circle_brightest_spot(image, brightest_spot, radius):
    image = image.copy()
    cv2.circle(image, brightest_spot, radius, (0, 0, 255), 4)
    cross_size = 5
    cross_thickness = 2
    cross_color = (0, 255, 0) 
    center_x, center_y = brightest_spot
    cv2.line(image, (center_x - cross_size, center_y), (center_x + cross_size, center_y), cross_color, cross_thickness)
    cv2.line(image, (center_x, center_y - cross_size), (center_x, center_y + cross_size), cross_color, cross_thickness)

    
    return image

def mask_outside_circle(image, brightest_spot, radius):
    mask = np.zeros_like(image)
    cv2.circle(mask, brightest_spot, radius, (255, 255, 255), -1)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


images,names = read_images_from_folders("Assignment-2/Fundus-image")
radius = 50
images = [image for image,name in zip(images,names) if "1ffa" in name] 
names = [name for name in names if "1ffa" in name]

# images = images[:1]
# names = names[:1]

for path,name in zip(images,names):
    image = resize_image(cv2.imread(path))
    vessels = extract_vessels(image)

    # brightest_spot = find_brightest_spot(image)
    # print(name ,"image Coordinates:", brightest_spot)
        
    # image = circle_brightest_spot(image, brightest_spot, radius)
    # image_circle = mask_outside_circle(image, brightest_spot,radius)
    # vessels_circle = mask_outside_circle(vessels, brightest_spot,radius)

    # cv2.imshow(name, image)
    # cv2.imshow(name + ' nerves', vessels)
    # cv2.imshow(name + ' inside circle', image_circle)
    # cv2.imshow(name + ' nerves inside circle', vessels_circle)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()   

    brightest_spots = find_brightest_spots(image)
    for spot in brightest_spots:
        image = circle_brightest_spot(image, spot, radius)
        image_circle = mask_outside_circle(image, spot, radius)
        vessels_circle = mask_outside_circle(vessels, spot, radius)
        cv2.imshow(name, image)
        cv2.imshow(name + ' nerves', vessels)
        cv2.imshow(name + ' inside circle', image_circle)
        cv2.imshow(name + ' nerves inside circle', vessels_circle)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

