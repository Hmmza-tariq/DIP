import os
import cv2
import numpy as np
import pickle

# Function to extract HOG features from an image
def extract_hog_features(image, resize_to=(64, 64)):
    """
    Extracts HOG features from an image after resizing it to the specified size.

    Args:
        image: The input image as a NumPy array.
        resize_to: A tuple representing the desired size for image resizing (optional, defaults to (64, 64)).

    Returns:
        A flattened NumPy array of HOG features.
    """
    image = cv2.resize(image, resize_to)  # Resize the image before feature extraction
    winSize = (64, 64)
    blockSize = (16, 16)
    blockStride = (8, 8)
    cellSize = (8, 8)
    nbins = 9

    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
    hog_features = hog.compute(image)
    return hog_features.flatten()

# Function to load images, remove background, and labels from a directory
def load_data(directory, resize_to=(128, 128), remove_background=True):
    labels = []
    images = []
    for subdir, _, files in os.walk(directory):
        for file in files:
            filepath = os.path.join(subdir, file)
            image = cv2.imread(filepath)

            if image is not None:
                if remove_background:
                    # Convert image to grayscale for background subtraction
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                    # Apply background subtraction
                    _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)

                    # Apply the mask to the image
                    image = cv2.bitwise_and(image, image, mask=mask)
                    cv2.imshow("he",image)
                    cv2.waitKey()

                # Resize the image
                image = cv2.resize(image, resize_to)

                label = os.path.basename(subdir)
                labels.append(label)
                images.append(image)
            else:
                print(f"Failed to load image: {filepath}")
    return images, labels

# Train folder directory
train_dir = 'cluster'  # Update with the correct path

# Load images and labels from the train folder with background removal and resizing
train_images, train_labels = load_data(train_dir, resize_to=(128, 128), remove_background=False)

# Compute HOG features for each image (using resized images)
hog_features_list = []
for image in train_images:
    hog_features = extract_hog_features(image)
    hog_features_list.append(hog_features)

# Save HOG features and labels into a file
hog_data = {'hog_features': hog_features_list, 'labels': train_labels}
with open('Project\example\hog_data.pkl', 'wb') as file:
    pickle.dump(hog_data, file)

print("HOG features and labels saved successfully!")


