import os
import cv2
import numpy as np

base_dir = 'D:/Semester 6/DIP/DIP-code/Project' 
train_dir = os.path.join(base_dir, 'train')
annotations_dir = os.path.join(train_dir, 'annotations')

# Function to segment images
def segment_images(annotation_path, train_image_path, output_path):
    annotation = cv2.imread(annotation_path, cv2.IMREAD_GRAYSCALE)
    train_image = cv2.imread(train_image_path)

    if annotation is None:
        raise ValueError(f"Failed to read annotation image: {annotation_path}")
    if train_image is None:
        raise ValueError(f"Failed to read train image: {train_image_path}")

    # print(f"Annotation shape: {annotation.shape}, Train image shape: {train_image.shape}")

    # Ensure the dimensions match
    if annotation.shape != train_image.shape[:2]:
        # print(f"Resizing train image from {train_image.shape[:2]} to {annotation.shape}")
        train_image = cv2.resize(train_image, (annotation.shape[1], annotation.shape[0]))

    # Create a mask for non-zero pixels
    mask = annotation > 0

    # Segment the image using the mask
    segmented_image = np.zeros_like(train_image)
    segmented_image[mask] = train_image[mask]
    cv2.imwrite(output_path, segmented_image)

# Iterate through each object class
object_classes = ['gun', 'knife'] 

for object_class in object_classes:
    annotation_class_dir = os.path.join(annotations_dir, object_class)
    train_class_dir = os.path.join(train_dir, object_class)
    output_class_dir = os.path.join(train_dir, f'segmented_{object_class}')
    os.makedirs(output_class_dir, exist_ok=True)

    annotation_files = os.listdir(annotation_class_dir)
    
    for annotation_file in annotation_files:
        annotation_path = os.path.join(annotation_class_dir, annotation_file)
        train_image_path = os.path.join(train_class_dir, annotation_file)
        output_path = os.path.join(output_class_dir, annotation_file)

        if os.path.exists(train_image_path):
            try:
                segment_images(annotation_path, train_image_path, output_path)
            except ValueError as e:
                print(f"Error processing {annotation_file}: {e}")
        else:
            print(f"Train image for {annotation_file} not found.")

print("Segmentation completed.")
