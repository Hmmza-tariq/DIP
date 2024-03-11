import cv2
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
import traceback
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

def display_histogram(v_set):
    plt.figure(figsize=(10, 8))
    i = 1
    for layer_code, layer_values in v_set.items():
        for j in range(3):
            plt.subplot(5, 3, i)
            plt.hist(layer_values[:, j], bins=256, range=(0, 256), color=['b', 'g', 'r'][j], alpha=0.5)
            plt.title(f"{layer_code} - {'BGR'[j]}")
            plt.ylim(0, 500)
            plt.xlim(0, 256)
            i += 1
        
    plt.tight_layout()
    plt.show()



def cca(original_images, mask_images):
    try:
        i = 0
        for img, mask in zip(original_images, mask_images):
            i += 1
            if img is None or mask is None:
                raise ValueError("One or both of the images could not be loaded.")

            color_codes = {
                (255, 172, 255): 'DEJ',  
                (0, 255, 190): 'DRM',  
                (160, 48, 112): 'EPI',  
                (224, 224, 224): 'KER',  
                (0, 0, 0): 'BKG'  
            }

            v_set = {
                'DEJ': [], 'DRM': [], 'EPI': [], 'KER': [], 'BKG': []
            }
            
            segmented_images = {}
            segmented_masks = {}

            for x in range(mask.shape[0]):
                for y in range(mask.shape[1]):
                    pixel = tuple(mask[x, y])
                    intensity = img[x, y]
                    for color, layer in color_codes.items():
                        if pixel == color:
                            v_set[layer].append(intensity)
                            if layer not in segmented_images:
                                segmented_images[layer] = np.zeros_like(img)
                                segmented_masks[layer] = np.zeros_like(mask)
                            segmented_images[layer][x, y] = img[x, y]
                            segmented_masks[layer][x, y] = mask[x, y]
                            break

            for layer, values in v_set.items():
                v_set[layer] = np.unique(values, axis=0)

            print('Segmented and Masked Images: ', i)

        for layer, values in v_set.items():
                v_set[layer] = np.unique(values, axis=0)


        img = cv2.imread("Assignment-1\Train\Tissue\RA23-01882-A1-1-PAS.[1536x2560].jpg", cv2.IMREAD_COLOR)
        mask = cv2.imread("Assignment-1\Train\Mask\RA23-01882-A1-1.[1536x2560].png", cv2.IMREAD_COLOR)
        
        rebuild_segmented_images = {layer: np.zeros_like(img) for layer in v_set}
        rebuild_segmented_masks = {layer: np.zeros_like(mask) for layer in v_set}


        for x in range(img.shape[0]):
            for y in range(img.shape[1]):
                img_intensity = img[x, y]
                for layer, intensities in v_set.items():
                    if img_intensity in intensities:
                        if layer in rebuild_segmented_images:
                            rebuild_segmented_images[layer][x, y] = img[x, y]
                        else:
                            rebuild_segmented_images[layer] = np.zeros_like(img)
                        break

        for x in range(mask.shape[0]):
            for y in range(mask.shape[1]):
                mask_intensity = mask[x, y]
                for layer, intensities in v_set.items():
                    if mask_intensity in intensities:
                        if layer in rebuild_segmented_masks:
                            rebuild_segmented_masks[layer][x, y] = mask[x, y]
                        else:
                            rebuild_segmented_masks[layer] = np.zeros_like(mask)                          
                        break

        return v_set, segmented_images, segmented_masks, rebuild_segmented_images, rebuild_segmented_masks

    except Exception as e:
        traceback.print_exc()  
        return None, None, None

# original_image_path = 'Assignment-1\Train\Tissue\RA23-01882-A1-1-PAS.[1536x2560].jpg'
# mask_image_path = 'Assignment-1\Train\Mask\RA23-01882-A1-1.[1536x2560].png'
# Define the paths to the tissue and mask folders
tissue_folder_path = 'Assignment-1/Train/Tissue/'
mask_folder_path = 'Assignment-1/Train/Mask/'

original_images, mask_images = read_images_from_folders(tissue_folder_path, mask_folder_path)

v_set, segmented_images, segmented_masks, rebuild_segmented_images, rebuild_segmented_masks = cca(original_images, mask_images)

if v_set is not None and segmented_images is not None:
    # print(v_set)
    display_histogram(v_set)
    segmented_images_list = []
    segmented_masks_list = []
    rebuild_segmented_images_list = []
    rebuild_segmented_masks_list = []

    for layer, segmented_image in segmented_images.items():     
        segmented_images_list.append(segmented_image)       
        rebuild_segmented_image = rebuild_segmented_images.get(layer, np.zeros_like(segmented_image))        
        rebuild_segmented_images_list.append(rebuild_segmented_image)
        segmented_mask = segmented_masks.get(layer, np.zeros_like(segmented_image))        
        segmented_masks_list.append(segmented_mask)
        rebuild_segmented_mask = rebuild_segmented_masks.get(layer, np.zeros_like(segmented_image))        
        rebuild_segmented_masks_list.append(rebuild_segmented_mask)
   
    segmented_images_combined = np.hstack(segmented_images_list)
    segmented_masks_combined = np.hstack(segmented_masks_list)
    rebuild_segmented_images_combined = np.hstack(rebuild_segmented_images_list)
    rebuild_segmented_masks_combined = np.hstack(rebuild_segmented_masks_list)
   
    grid_image = np.vstack((segmented_images_combined, segmented_masks_combined, rebuild_segmented_images_combined, rebuild_segmented_masks_combined))

    
    screen_height, screen_width = np.array(grid_image.shape[:2]) * 0.4 

    if grid_image.shape[0] > screen_height or grid_image.shape[1] > screen_width:
        scale_factor = min(screen_height / grid_image.shape[0], screen_width / grid_image.shape[1])
        grid_image = cv2.resize(grid_image, (0, 0), fx=scale_factor, fy=scale_factor)

    cv2.imshow("Segmented and Masked Images", grid_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    