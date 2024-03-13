import cv2
import numpy as np
import os

def read_images_from_folders(tissue_folder, mask_folder):
    original_images = []
    mask_images = []
    tissue_files = os.listdir(tissue_folder)
    tissue_files.sort()
    mask_files = os.listdir(mask_folder)
    mask_files.sort()
    for tissue_file, mask_file in zip(tissue_files, mask_files):
        tissue_path = os.path.join(tissue_folder, tissue_file)
        mask_path = os.path.join(mask_folder, mask_file)
        tissue_img = cv2.imread(tissue_path, cv2.IMREAD_COLOR)
        mask_img = cv2.imread(mask_path, cv2.IMREAD_COLOR)
        original_images.append(tissue_img)
        mask_images.append(mask_img)
    return original_images, mask_images

def remove_background(image, mask):
    try:
        if image is None or mask is None:
            raise ValueError("One or both of the images could not be loaded.")
        mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_gray, connectivity=8)
        largest_label = 1
        max_area = stats[largest_label, cv2.CC_STAT_AREA]
        for label in range(2, num_labels):
            area = stats[label, cv2.CC_STAT_AREA]
            if area > max_area:
                largest_label = label
                max_area = area
        largest_component_mask = (labels == largest_label).astype(np.uint8)
        result_image = cv2.bitwise_and(image, image, mask=largest_component_mask)
        return result_image
    except Exception as e:
        print(f"Error: {e}")
        return None



original_images, mask_images = read_images_from_folders('Assignment-1/Test/Tissue/', 'Assignment-1/Test/Mask/')
for img, mask in zip(original_images, mask_images):
    result = remove_background(img, mask)
    grid_image = np.hstack([img, mask, result])
    cv2.imshow('Original', grid_image)
    cv2.waitKey(0)


cv2.waitKey(0)

