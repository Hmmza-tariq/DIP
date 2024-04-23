import cv2
import numpy as np
import os

def read_images_from_folders(folder_path):
    images = []
    names = []
    files = os.listdir(folder_path)    
    files.sort()
    for file in files:       
        path = os.path.join(folder_path, file)
        names.append(file)
        images.append(path)
    return images,names


def find_brightest_spot(image):
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(grayscale_image)
    brightest_spot = max_loc
    return brightest_spot

images,names = read_images_from_folders("Assignment-2/Fundus-image")
for path,name in zip(images,names):
    image = cv2.imread(path)
    brightest_spot = find_brightest_spot(image)
    print(name ,"Coordinates:", brightest_spot)
