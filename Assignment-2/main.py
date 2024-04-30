import os
import cv2
import numpy as np
from matplotlib.image import imread
import matplotlib.pyplot as plt

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

def pre_process_image(image, new_size=(512, 512)):
    resized_image = cv2.resize(image, new_size)
    return resized_image

def extract_vessels(image, s=2500):
    structure_element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    min_component_area = s
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur_image = cv2.GaussianBlur(grayscale_image, (5, 5), 0)
    edges = cv2.Canny(blur_image, 30, 10)
    connected_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, structure_element)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(connected_edges, connectivity=8, ltype=cv2.CV_32S)
    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] < min_component_area:
            connected_edges[labels == label] = 0
    output = cv2.erode(cv2.dilate(connected_edges, structure_element), structure_element)
    num_vessels = cv2.countNonZero(connected_edges)
    # print("Number of vessels found:", num_vessels)
    if num_vessels == 0:
        print("No vessels found")
        output = extract_vessels(image, s-2000)
    return output

def find_brightest_spots(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    max_intensity = np.amax(blurred_image)
    _, thresholded_image = cv2.threshold(blurred_image, max_intensity * 0.7, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresholded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:20]
    
    bright_spots = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        center_x = x + w // 2
        center_y = y + h // 2
        bright_spots.append((center_x, center_y))
    
    return bright_spots

def find_intersection_point(image):
    structure_element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    closed_image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, structure_element)
    eroded_image = cv2.erode(closed_image, structure_element)
    contours, _ = cv2.findContours(eroded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            center_x = int(M["m10"] / M["m00"])
            center_y = int(M["m01"] / M["m00"])
            intersection_point = (center_x, center_y)
            return intersection_point
    return None

def find_optic_disk(image, brightest_spots,intersection_point, radius):
    max_vessel_count = -1
    best_spot = None
    for spot in brightest_spots:
        masked_image = mask_circle(image, spot, radius)
        vessel_count = cv2.countNonZero(masked_image)
        distance = np.sqrt((spot[0] - intersection_point[0])**2 + (spot[1] - intersection_point[1])**2)
        if distance < radius:
            return intersection_point
        if vessel_count > max_vessel_count:
            max_vessel_count = vessel_count
            best_spot = spot
    return best_spot 

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

def mask_circle(image, brightest_spot, radius):
    mask = np.zeros_like(image)
    cv2.circle(mask, brightest_spot, radius, (255, 255, 255), -1)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def draw_brightest_spots(image, brightest_spots):
    for spot in brightest_spots:
        cv2.circle(image, spot, 5, (0, 255, 0), -1)

def display(images, figure_name='Figure'):
    plt.figure(figsize=(16, 4))
    for i in range(len(images)):
        plt.subplot(1, len(images), i+1)
        plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB), aspect='equal')
        if i == 0:
            plt.title('Brightest Spots')
        elif i == 1:
            plt.title('Vessels')
        elif i == 2:
            plt.title('Intersection Point')
        else:
            plt.title('Result Image')
        plt.suptitle(figure_name, fontsize=16)
        plt.axis('off')
    plt.tight_layout()  
    plt.show()


images,names = read_images_from_folders("Assignment-2/Fundus-image")
radius = 50

# images = [image for image,name in zip(images,names) if "test" in name] 
# names = [name for name in names if "test" in name]

# images = [image for image,name in zip(images,names) if "1ffa" in name]            
# names = [name for name in names if "1ffa" in name]

# images = images[:1]
# names = names[:1]

for path,name in zip(images,names):
    image = pre_process_image(cv2.imread(path))
    intersection_point_image = image.copy()
    image_with_spots = image.copy()
    vessels = extract_vessels(image)
    brightest_spots = find_brightest_spots(image)
    vessels = extract_vessels(image)
    intersection_point = find_intersection_point(vessels)
    if intersection_point:
        cv2.circle(intersection_point_image, intersection_point, 5, (255, 0, 0), -1)    
    best_spot = find_optic_disk(vessels, brightest_spots,intersection_point, radius)
    result_image = circle_brightest_spot(image, best_spot, radius)
    draw_brightest_spots(image_with_spots, brightest_spots)
    display([image_with_spots,vessels,intersection_point_image,result_image],name)


