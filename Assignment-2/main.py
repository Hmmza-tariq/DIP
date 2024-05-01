import os
import cv2
import csv
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

def read_image_centers(file_path):
    centers = []
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line:  
                parts = line.split(',')
                if len(parts) == 3: 
                    x = int(parts[1].strip())
                    y = int(parts[2].strip())
                    centers.append((x, y))
    return  centers

def pre_process_image(image, new_size=(512, 512)):
    original_size = image.shape[:2]
    target_size = new_size 
    scaling_factor = (target_size[1] / original_size[1], target_size[0] / original_size[0]) 
    resized_image = cv2.resize(image, new_size)
    return resized_image, scaling_factor

def extract_vessels(image, min_component_area=2500):
    structure_element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur_image = cv2.GaussianBlur(grayscale_image, (5, 5), 0)
    edges = cv2.Canny(blur_image, 30, 20)
    connected_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, structure_element)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(connected_edges, connectivity=8, ltype=cv2.CV_32S)
    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] < min_component_area:
            connected_edges[labels == label] = 0
    output = connected_edges
    num_vessels = cv2.countNonZero(connected_edges)
    # print('number of vessels found: ', num_vessels)
    if num_vessels < 6000:
        if min_component_area < 500:
            # print('less vessels found, s < 500 ')
            return output,num_vessels
        elif min_component_area < 1100:
            # print('less vessels found, s < 1100')
            output,num_vessels = extract_vessels(image, min_component_area-500)
        else:
            # print('less vessels found, s = 2500')
            output,num_vessels = extract_vessels(image, min_component_area-1500)

    return output, num_vessels

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

def analyze_data(image, brightest_spots,intersection_point,count, radius):
    max_vessel_count = -1
    list_of_info = []
    skip = False
    for spot in brightest_spots:
        info = {
            "center": (0, 0),
            "intensity": 0,
            "vessel_count": 0,
            "distance_from_intersection": 0,
            "neighbor_spots": 0,
        }

        masked_image = mask_circle(image, spot, radius)
        center = spot
        intensity = np.mean(masked_image)
        vessel_count = cv2.countNonZero(masked_image)
        # print("Vessel Count: ",vessel_count, "Count: ", count)
        distance_from_intersection = float('inf')
        neighbor_spots = 0;

        if intersection_point is not None:
            distance_from_intersection = np.sqrt((spot[0] - intersection_point[0])**2 + (spot[1] - intersection_point[1])**2)
        if distance_from_intersection < radius:

            for bright_spot in brightest_spots:
                if bright_spot != spot:
                    distance = float('inf')
                    if intersection_point is not None:
                        distance = np.sqrt((bright_spot[0] - intersection_point[0])**2 + (bright_spot[1] - intersection_point[1])**2)
                    if distance < radius:
                        neighbor_spots += 1

            if neighbor_spots > len(brightest_spots)/2.5:
                center = intersection_point
                skip = True

        if skip is not True:
            if vessel_count > max_vessel_count:
                max_vessel_count = vessel_count

        info["center"] = center
        info["intensity"] = intensity
        info["vessel_count"] = vessel_count
        info["distance_from_intersection"] = distance_from_intersection
        info["neighbor_spots"] = neighbor_spots
        list_of_info.append(info)
    return  list_of_info 

def find_best_spot(info):
    info = sorted(info, key=lambda x: (x["vessel_count"], x['intensity'], x["distance_from_intersection"], x['neighbor_spots']), reverse=True)
    return info[0]["center"],info

def circle_brightest_spot(image, brightest_spot, radius):
    image = image.copy()
    cv2.circle(image, brightest_spot, radius, (0, 0, 255), 4)
    cross_size = int(radius/10)
    cross_thickness = int(radius/25)
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

def display(images, title,name, save_path):
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
        elif i == 3:
            plt.title('Derived Result Image')
        else:
            plt.title('Actual Result Image')
        plt.suptitle(title, fontsize=16)
        plt.axis('off')
    plt.tight_layout()  
    plt.savefig(save_path + "/" + name + ".png" )
    plt.show()

def print_info_table(list_of_info):
    print("Optic Disk Information:")
    print("{:<15} {:<15} {:<15} {:<15} {:<15} ".format("Center", "Intensity", "Vessel Count", "Dst from IP", "Nbr Spots"))
    for info in list_of_info:
        center = info["center"]
        intensity =  round(info["intensity"], 2)
        vessel_count = info["vessel_count"]
        distance_from_intersection = round(info["distance_from_intersection"])
        neighbor_spots = info["neighbor_spots"]

        center_str = f"({center[0]}, {center[1]})" 
        print("{:<15} {:<15} {:<15} {:<15} {:<15} ".format(center_str,intensity, vessel_count, distance_from_intersection, neighbor_spots,))
    print('----------------------------------------------------')


images,names = read_images_from_folders("Assignment-2/Fundus-image")
actual_centers = read_image_centers("Assignment-2/optic_disc_centres.csv")
derived_centers = []
distances = []
radius = 50
# images = [image for image,name in zip(images,names) if "test" in name] 
# names = [name for name in names if "test" in name]

# images = [image for image,name in zip(images,names) if "1ffa" in name]            
# names = [name for name in names if "1ffa" in name]

# images = images[:1]
# names = names[:1]

for path,name,center in zip(images,names,actual_centers):
    image = cv2.imread(path)
    preprocessed_image,scaling_factor = pre_process_image(image)
    intersection_point_image = preprocessed_image.copy()
    image_with_spots = preprocessed_image.copy()
    vessels,num_vessels = extract_vessels(preprocessed_image)
    brightest_spots = find_brightest_spots(preprocessed_image)
    intersection_point = find_intersection_point(vessels)
    if intersection_point:
        cv2.circle(intersection_point_image, intersection_point, 5, (255, 0, 0), -1)    
    info = analyze_data(vessels, brightest_spots,intersection_point,num_vessels, radius)
    best_spot,info = find_best_spot(info)
    derived_result_image = circle_brightest_spot(preprocessed_image, best_spot, radius)
    draw_brightest_spots(image_with_spots, brightest_spots)
    best_spot_scaled = (int( best_spot[0]/scaling_factor[0] ), int( best_spot[1]/scaling_factor[1] ))  
    actual_radius = radius
    if center[0] > 1000:
        actual_radius = int(radius * (center[0]/300))
    actual_result_image = circle_brightest_spot(image, center, actual_radius)
    distance = round(np.sqrt((best_spot_scaled[0] - center[0])**2 + (best_spot_scaled[1] - center[1])**2),2)
    distances.append(distance)
    derived_centers.append(best_spot_scaled)
    display([image_with_spots, vessels, intersection_point_image, derived_result_image,actual_result_image],'Name: ' +  name + '\nDerived center: ' + str(best_spot_scaled) + '\nActual center: ' + str(center) + '\nDistance: ' + str(distance),name,"Assignment-2/Result" )
    print_info_table(info)

print("Distances:")
print("{:<20} {:<20} {:<20} {:<20}".format("Image", "Derived OC", "Actual OC", "Distance"))
for (name, derived_center, actual_center, distance) in zip(names,derived_centers, actual_centers,distances):
    print("{:<20} {:<20} {:<20} {:<20}".format(name, str(derived_center), str(actual_center) , str(distance)))
with open("Assignment-2/resultant_optic_disc_centres.csv", mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Image", "Derived OC", "Actual OC", "Distance"])
    for name, derived_center, actual_center, distance in zip(names, derived_centers, actual_centers, distances):
        writer.writerow([name, str(derived_center), str(actual_center), str(distance)])
print("Results have been written to", "Assignment-2/resultant_optic_disc_centres.csv")