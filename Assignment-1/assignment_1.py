import cv2
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
import traceback

def display_outputs(original_image, skin_region, layer_groups, dice_scores):
    
    plt.figure(figsize=(10, 8))
    plt.subplot(3, 3, 1)
    plt.imshow(cv2.cvtColor(cv2.imread(original_image), cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')

    
    plt.subplot(3, 3, 2)
    plt.imshow(skin_region, cmap='gray')
    plt.title('Skin Region')
    plt.axis('off')

    
    for i, (layer_code, layer_masks) in enumerate(layer_groups.items(), start=3):
        plt.subplot(3, 3, i)
        if layer_masks:  
            plt.imshow(layer_masks, cmap='gray')
            plt.title(layer_code)
        else:
            plt.title(f"No {layer_code} mask")
        plt.axis('off')

    
    plt.subplot(3, 1, 3)
    dice_keys = []
    dice_values = []
    for layer_code, dice_coefficient in dice_scores.items():
        if dice_coefficient is not None:
            dice_keys.append(layer_code)
            dice_values.append(dice_coefficient)
    plt.bar(dice_keys, dice_values)
    plt.title('Dice Coefficients')
    plt.xlabel('Skin Layers')
    plt.ylabel('Dice Coefficient')

    plt.tight_layout()
    plt.show()

import cv2
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

def display_histogram(v_set):
    plt.figure(figsize=(10, 8))
    i = 1
    for layer_code, layer_values in v_set.items():
        plt.subplot(3, 2, i)  
        plt.hist(layer_values.flatten(), bins=20, color='b', alpha=0.7)
        plt.title(f'{layer_code} Histogram')
        plt.xlabel('Pixel Intensity')
        plt.ylabel('Frequency')
        i += 1
    plt.tight_layout()
    plt.show()

def cca(original_image, mask_image):
    try:
        img = cv2.imread(original_image, cv2.IMREAD_COLOR)
        mask = cv2.imread(mask_image, cv2.IMREAD_COLOR)

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

        for layer, intensities in v_set.items():
            v_set[layer] = np.array(intensities)

        for layer, intensities in v_set.items():
            v_set[layer] = np.unique(intensities, axis=0)

        rebuild_segmented_images = {}
        rebuild_segmented_masks = {}
        for layer, intensities in v_set.items():
            rebuild_segmented_images[layer] = np.zeros_like(img)
            rebuild_segmented_masks[layer] = np.zeros_like(mask)
            for x in range(img.shape[0]):
                for y in range(img.shape[1]):
                    if (img[x, y] == intensities).all(axis=-1).any():
                        rebuild_segmented_images[layer][x, y] = img[x, y]

            for x in range(mask.shape[0]):
                for y in range(mask.shape[1]):
                    if (mask[x, y] == intensities).all(axis=-1).any():
                        rebuild_segmented_masks[layer][x, y] = mask[x, y]

        return v_set, segmented_images, segmented_masks, rebuild_segmented_images, rebuild_segmented_masks

    except Exception as e:
        print(e)
        return None, None, None, None, None

original_image_path = 'Assignment-1\Train\Tissue\RA23-01882-A1-1-PAS.[1536x2560].jpg'
mask_image_path = 'Assignment-1\Train\Mask\RA23-01882-A1-1.[1536x2560].png'

v_set, segmented_images, segmented_masks, rebuild_segmented_images, rebuild_segmented_masks = cca(original_image_path, mask_image_path)

if v_set is not None and segmented_images is not None:
    print(tabulate(v_set, headers='keys', tablefmt='pretty'))

    segmented_images_list = []
    segmented_masks_list = []
    rebuild_segmented_images_list = []
    rebuild_segmented_masks_list = []

    for layer, segmented_image in segmented_images.items():     
        segmented_images_list.append(segmented_image)       
        rebuild_segmented_images_list.append(rebuild_segmented_images.get(layer, np.zeros_like(segmented_image)))
        segmented_masks_list.append(segmented_masks.get(layer, np.zeros_like(segmented_image)))
        rebuild_segmented_masks_list.append(rebuild_segmented_masks.get(layer, np.zeros_like(segmented_image)))
   
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