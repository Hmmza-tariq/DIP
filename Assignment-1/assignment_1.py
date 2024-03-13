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

def refine_layer(v_set, code, min_intensity, max_intensity):
    layer_values = []
    try:
        for i in range(len(v_set[code])):  
            for j in range(len(v_set[code][i])):  
              if np.all(v_set[code][i] >= min_intensity) and np.all(v_set[code][i] <= max_intensity):
                layer_values.append(v_set[code][i])
                break

        v_set[code] = np.array(layer_values)       
        display_histograms(v_set)
    except Exception as e:
        print(f"Error: {e}")
    return v_set
def input_refine_layer(v_set):
    try:
        img = cv2.imread('resultant_histogram.png')
        cv2.imshow('Resultant Histogram', img)
        for layer_code, layer_values in v_set.items():
            if layer_code == 'DEJ':
                min_b = 241
                max_b = 256
                min_g = 99
                max_g = 183
                min_r = 174
                max_r = 256
            elif layer_code == 'DRM':
                min_b = 197
                max_b = 249
                min_g = 90
                max_g = 214
                min_r = 171
                max_r = 249
            elif layer_code == 'EPI':
                min_b = 214
                max_b = 242
                min_g = 25
                max_g = 90
                min_r = 113
                max_r = 182
            elif layer_code == 'KER':
                min_b = 200
                max_b = 256
                min_g = 65
                max_g = 95
                min_r = 78
                max_r = 135
            v_set = refine_layer(v_set, layer_code, [min_b, min_g, min_r], [max_b, max_g, max_r])
        cv2.destroyAllWindows()
        return v_set
    except Exception as e:
        print(f"Error: {e}")
        return None
    
def display_histograms(v_set):
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
    plt.savefig('histograms.png')
    plt.show()
    
def display_resultant_histogram(v_set):
    plt.figure(figsize=(15, 5))
    layer_colors = ['b', 'g', 'r', 'm','k', 'orange', 'lime', 'pink']
    num_channels = 3
    for j in range(num_channels):
        plt.subplot(1, num_channels, j+1)
        for idx, (layer_code, layer_values) in enumerate(v_set.items()):
            color_idx = idx % len(layer_colors)
            plt.hist(layer_values[:, j], bins=256, range=(0, 256), color=layer_colors[color_idx], alpha=0.5, label=layer_code)
        plt.title(f"{'BGR'[j]} Channel")
        plt.ylim(0, 1000)
        plt.xlim(0, 256)
        plt.legend()
    plt.tight_layout()
    plt.savefig('resultant_histogram.png')
    plt.show()
    return v_set

def display_images(segmented_images, segmented_masks, rebuild_segmented_images, rebuild_segmented_masks):
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

def print_table(v_set):
    table = []
    for layer, values in v_set.items():
        table.append([layer, len(values)])
    print(tabulate(table, headers=["Layer", "Unique Intensities"]))

def cca(images, masks):
    try:
        i = 0
        for image, mask in zip(images, masks):
            i += 1
            image = remove_background(image,mask)
            if image is None or mask is None:
                raise ValueError("One or both of the images could not be loaded.")

            color_codes = {
                (255, 172, 255): 'DEJ',  
                (0, 255, 190): 'DRM',  
                (160, 48, 112): 'EPI',  
                (224, 224, 224): 'KER',  
                (0, 0, 0): 'BKG'  
            }

            v_set = {
                'DEJ': [], 'DRM': [], 'EPI': [], 'KER': []
            }
            
            segmented_images = {}
            segmented_masks = {}

            for x in range(mask.shape[0]):
                for y in range(mask.shape[1]):
                    pixel = tuple(mask[x, y])
                    intensity = image[x, y]
                    for color, layer in color_codes.items():
                        if layer == 'BKG':
                            continue
                        elif pixel == color:
                            v_set[layer].append(intensity)
                            if layer not in segmented_images:
                                segmented_images[layer] = np.zeros_like(image)
                                segmented_masks[layer] = np.zeros_like(mask)
                            segmented_images[layer][x, y] = image[x, y]
                            segmented_masks[layer][x, y] = mask[x, y]
                            break

            for layer, values in v_set.items():
                v_set[layer] = np.unique(values, axis=0)

            print('Segmented and Masked Images: ', i)

        for layer, values in v_set.items():
                v_set[layer] = np.unique(values, axis=0)



        return v_set, segmented_images, segmented_masks

    except Exception as e:
        traceback.print_exc()  
        return None, None, None

def rebuild_images(v_set, images, masks):
    i = 0
    try:
        for image, mask in zip(images, masks):
            i += 1
            image = remove_background(image, mask)
            rebuild_segmented_images = {layer: np.zeros_like(image) for layer in v_set}
            rebuild_segmented_masks = {layer: np.zeros_like(mask) for layer in v_set}
            for x in range(image.shape[0]):
                for y in range(image.shape[1]):
                    img_intensity = image[x, y]
                    for layer, intensities in v_set.items():
                        if np.any(img_intensity == intensities):
                            if layer in rebuild_segmented_images:
                                rebuild_segmented_images[layer][x, y] = image[x, y]
                            else:
                                rebuild_segmented_images[layer] = np.zeros_like(image)
                            break
            
            for x in range(mask.shape[0]):
                for y in range(mask.shape[1]):
                    mask_intensity = mask[x, y]
                    for layer, intensities in v_set.items():
                        if  np.any(mask_intensity in intensities):
                            if layer in rebuild_segmented_masks:
                                rebuild_segmented_masks[layer][x, y] = mask[x, y]
                            else:
                                rebuild_segmented_masks[layer] = np.zeros_like(mask)                          
                            break
            print('Rebuilt Images: ', i)
            
        return rebuild_segmented_images, rebuild_segmented_masks
    
    except Exception as e:
        traceback.print_exc()  
        return None, None

train_images, train_masks = read_images_from_folders('Assignment-1/Train/Tissue/', 'Assignment-1/Train/Mask/')
test_images, test_masks = read_images_from_folders('Assignment-1/Train/Tissue/', 'Assignment-1/Train/Mask/')

train_images  = [train_images[0]] 
train_masks = [train_masks[0]]
test_images = [test_images[0]]
test_masks = [test_masks[0]]

v_set, segmented_images, segmented_masks = cca(train_images, train_masks)

display_histograms(v_set)
display_resultant_histogram(v_set)
v_set = input_refine_layer(v_set)
display_histograms(v_set)
display_resultant_histogram(v_set)
rebuild_segmented_images, rebuild_segmented_masks = rebuild_images(v_set, test_images, test_masks)

print_table(v_set)
display_images(segmented_images, segmented_masks, rebuild_segmented_images, rebuild_segmented_masks)


    