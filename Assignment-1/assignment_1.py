import cv2
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
import traceback
import os
import ipywidgets as widgets
from IPython.display import display

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

def display_histograms(v_set):

    def update_histogram(b_range, g_range, r_range):
            plt.figure(figsize=(15, 5))
            layer_colors = ['b', 'g', 'r', 'm']
            num_layers = len(v_set)
            num_channels = 3
            for j in range(num_channels):
                plt.subplot(1, num_channels, j+1)
                for idx, (layer_code, layer_values) in enumerate(v_set.items()):
                    color_idx = idx % len(layer_colors)
                    b_mask = np.logical_and(layer_values[:, 0] >= b_range[0], layer_values[:, 0] <= b_range[1])
                    g_mask = np.logical_and(layer_values[:, 1] >= g_range[0], layer_values[:, 1] <= g_range[1])
                    r_mask = np.logical_and(layer_values[:, 2] >= r_range[0], layer_values[:, 2] <= r_range[1])
                    mask = np.logical_and(np.logical_and(b_mask, g_mask), r_mask)
                    plt.hist(layer_values[mask, j], bins=256, range=(0, 256), color=layer_colors[color_idx], alpha=0.5, label=layer_code)
                plt.title(f"{'BGR'[j]} Channel")
                plt.ylim(0, 800)
                plt.xlim(0, 256)
                plt.legend()
            plt.tight_layout()
            plt.show()
    
    b_slider = widgets.FloatRangeSlider(value=[0, 255], min=0, max=255, step=1, description='B Range:', continuous_update=False)
    g_slider = widgets.FloatRangeSlider(value=[0, 255], min=0, max=255, step=1, description='G Range:', continuous_update=False)
    r_slider = widgets.FloatRangeSlider(value=[0, 255], min=0, max=255, step=1, description='R Range:', continuous_update=False)
    
    widgets.interactive(update_histogram, b_range=b_slider, g_range=g_slider, r_range=r_slider)

def display_resultant_histogram(v_set):
    plt.figure(figsize=(15, 5))
    layer_colors = ['b', 'g', 'r', 'm', 'y', 'k', 'orange', 'lime', 'pink']
    num_layers = len(v_set)
    num_channels = 3
    for j in range(num_channels):
        plt.subplot(1, num_channels, j+1)
        for idx, (layer_code, layer_values) in enumerate(v_set.items()):
            color_idx = idx % len(layer_colors)
            plt.hist(layer_values[:, j], bins=256, range=(0, 256), color=layer_colors[color_idx], alpha=0.5, label=layer_code)
        plt.title(f"{'BGR'[j]} Channel")
        plt.ylim(0, 800)
        plt.xlim(0, 256)
        plt.legend()
    plt.tight_layout()
    plt.show()

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

def cca(original_images, mask_images, test_img, test_mask):
    try:
        i = 0
        for img, mask in zip(original_images, mask_images):
            i += 1
            img = remove_background(img,mask)
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
                'DEJ': [], 'DRM': [], 'EPI': [], 'KER': []
            }
            
            segmented_images = {}
            segmented_masks = {}

            for x in range(mask.shape[0]):
                for y in range(mask.shape[1]):
                    pixel = tuple(mask[x, y])
                    intensity = img[x, y]
                    for color, layer in color_codes.items():
                        if layer == 'BKG':
                            continue
                        elif pixel == color:
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


        test_img = remove_background(test_img, test_mask)
        rebuild_segmented_images = {layer: np.zeros_like(test_img) for layer in v_set}
        rebuild_segmented_masks = {layer: np.zeros_like(test_mask) for layer in v_set}


        for x in range(test_img.shape[0]):
            for y in range(test_img.shape[1]):
                img_intensity = test_img[x, y]
                for layer, intensities in v_set.items():
                    if np.any(img_intensity == intensities):
                        if layer in rebuild_segmented_images:
                            rebuild_segmented_images[layer][x, y] = test_img[x, y]
                        else:
                            rebuild_segmented_images[layer] = np.zeros_like(test_img)
                        break

        for x in range(test_mask.shape[0]):
            for y in range(test_mask.shape[1]):
                mask_intensity = test_mask[x, y]
                for layer, intensities in v_set.items():
                    if  np.any(mask_intensity in intensities):
                        if layer in rebuild_segmented_masks:
                            rebuild_segmented_masks[layer][x, y] = test_mask[x, y]
                        else:
                            rebuild_segmented_masks[layer] = np.zeros_like(test_mask)                          
                        break

        return v_set, segmented_images, segmented_masks, rebuild_segmented_images, rebuild_segmented_masks

    except Exception as e:
        traceback.print_exc()  
        return None, None, None

original_images, mask_images = read_images_from_folders('Assignment-1/Train/Tissue/', 'Assignment-1/Train/Mask/')
original_images  = [ original_images[0]]
mask_images = [mask_images[0]]
test_images, test_masks = read_images_from_folders('Assignment-1/Train/Tissue/', 'Assignment-1/Train/Mask/')
test_img = test_images[0]
test_mask = test_masks[0]

v_set, segmented_images, segmented_masks, rebuild_segmented_images, rebuild_segmented_masks = cca(original_images, mask_images, test_img, test_mask)
display_histograms(v_set)
display_resultant_histogram(v_set)
print_table(v_set)
display_images(segmented_images, segmented_masks, rebuild_segmented_images, rebuild_segmented_masks)


    