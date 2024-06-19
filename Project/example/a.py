import os
import cv2
import numpy as np

# Paths to the folders
masks_folder = os.path.join('Project\\test', 'annotations', 'knife')
images_folder = os.path.join('Project\\test', 'knife')
output_folder = 'Project\example\cluster'

# Create the output folders if they don't exist
gun_folder = os.path.join(output_folder, 'gun')
knife_folder = os.path.join(output_folder, 'knife')
safe_folder = os.path.join(output_folder, 'safe')
os.makedirs(gun_folder, exist_ok=True)
os.makedirs(knife_folder, exist_ok=True)
os.makedirs(safe_folder, exist_ok=True)

# Get list of files in the masks folder
mask_files = [f for f in os.listdir(masks_folder) if os.path.isfile(os.path.join(masks_folder, f))]

for mask_file in mask_files:
    # Construct full file path for mask and image
    mask_path = os.path.join(masks_folder, mask_file)
    image_path = os.path.join(images_folder, mask_file)
    
    # Check if the corresponding image exists
    if not os.path.exists(image_path):
        print(f"Image corresponding to mask {mask_file} does not exist. Skipping...")
        continue

    # Read the mask and image
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # Read mask as grayscale
    image = cv2.imread(image_path)

    if mask is None or image is None:
        print(f"Failed to read mask or image for {mask_file}. Skipping...")
        continue

    # Resize the mask to match the image dimensions
    resized_mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Process each unique non-zero value in the mask
    for value in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        # Create a mask for the current value
        value_mask = resized_mask == value

        if np.any(value_mask):
            # Segment the image using the mask
            segmented_image = np.zeros_like(image)
            segmented_image[value_mask] = image[value_mask]

            # Determine the output folder based on the value
            if value in [1]:
                output_subfolder = gun_folder
            elif value in [2]:
                output_subfolder = knife_folder
            else:
                output_subfolder = safe_folder  # Default to safe_folder for unexpected values
            
            # Save the segmented image
            output_path = os.path.join(output_subfolder, mask_file)
            cv2.imwrite(output_path, segmented_image)

            print(f"Processed and saved segmented image for {mask_file} in folder {output_subfolder}")

print("Processing completed.")


