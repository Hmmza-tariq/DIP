
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
import pickle

# Function to read features from a .pkl file
def read_features_from_file(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data['hog_features'], data['labels']

# Function to extract HOG features from an image
def extract_hog_features(image, resize_to=(64, 64)):
    image = cv2.resize(image, resize_to)
    winSize = (64, 64)
    blockSize = (16, 16)
    blockStride = (8, 8)
    cellSize = (8, 8)
    nbins = 9

    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
    hog_features = hog.compute(image)
    return hog_features.flatten()

# Function to apply negative transformation
def negative_transformation(image):
    return 255 - image

# Function to apply contrast stretching
def contrast_stretching(image):
    min_val = np.min(image)
    max_val = np.max(image)
    stretched = (image - min_val) * (255 / (max_val - min_val))
    return stretched.astype(np.uint8)

# Function to load images and labels from a directory for testing
def load_test_data(directory, resize_to=(128, 128)):
    labels = []
    features = []
    images = []
    for subdir, _, files in os.walk(directory):
        for file in files:
            filepath = os.path.join(subdir, file)
            image = cv2.imread(filepath)
            if image is not None:
                # Resize the image
                image = cv2.resize(image, resize_to)
                # Apply contrast stretching
                processed_image = contrast_stretching(image)
                feature_vector = extract_hog_features(processed_image)
                features.append(feature_vector)
                labels.append(os.path.basename(subdir))
                images.append((image, processed_image))  # Save both original and processed images
            else:
                print(f"Failed to load image: {filepath}")
    return features, labels, images

# Function to segment object from image and make the rest black
def segment_object(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)
    
    # Apply morphological operations to remove small noises
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    cleaned = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv2.findContours(cleaned.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create a black image as a mask
    mask = np.zeros_like(gray)
    
    # Draw contours on the mask to highlight the object
    for c in contours:
        area = cv2.contourArea(c)
        if area > 100:  # Minimum contour area to be considered (adjust as needed)
            cv2.drawContours(mask, [c], -1, (255), -1)  # Fill contour with white (255)

    # Apply the mask to the original image to keep only the detected object
    segmented_image = cv2.bitwise_and(image, image, mask=mask)
    
    return segmented_image, gray, blurred, edged, mask, contours

# Load HOG features and labels from the .pkl file
file_path = 'Project\example\hog_data.pkl'  # Update with the correct path to your .pkl file
train_features, train_labels = read_features_from_file(file_path)

# Ensure all feature vectors have the same length
max_length = max(len(f) for f in train_features)
train_features = [np.pad(f, (0, max_length - len(f)), 'constant') for f in train_features]

# Convert to NumPy arrays
train_features = np.array(train_features)
train_labels = np.array(train_labels)

# Train your classification model using the new feature set
# Example: Use SVM classifier
clf = SVC(kernel='linear')
clf.fit(train_features, train_labels)

# Load test data
test_dir = 'test'  # Update with the correct path
test_features, test_labels, test_images = load_test_data(test_dir)

# Ensure all feature vectors have the same length
max_length = max(len(f) for f in test_features)
test_features = [np.pad(f, (0, max_length - len(f)), 'constant') for f in test_features]

# Convert to NumPy arrays
test_features = np.array(test_features)
test_labels = np.array(test_labels)

# Predict test data labels
predictions = clf.predict(test_features)

# Evaluate accuracy
accuracy = clf.score(test_features, test_labels)
print(f'Test Accuracy: {accuracy}')

# Compute confusion matrix
unique_labels = sorted(set(test_labels))
conf_matrix = np.zeros((len(unique_labels), len(unique_labels)), dtype=int)
label_to_index = {label: idx for idx, label in enumerate(unique_labels)}

for true_label, pred_label in zip(test_labels, predictions):
    true_idx = label_to_index[true_label]
    pred_idx = label_to_index[pred_label]
    conf_matrix[true_idx, pred_idx] += 1

print('Confusion Matrix')
print(conf_matrix)

# Plot confusion matrix using seaborn
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=unique_labels, yticklabels=unique_labels)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Process test images and segment objects if they are threats
final_dir = 'final-images'
if not os.path.exists(final_dir):
    os.makedirs(final_dir)

for i, ((original_image, processed_image), label, prediction) in enumerate(zip(test_images, test_labels, predictions)):
    if prediction != "safe":
        segmented_image, gray, blurred, edged, mask, contours = segment_object(processed_image)
        
        # Debugging outputs
        print(f"Image {i}: Label={label}, Prediction={prediction}")
        print(f"Number of contours detected: {len(contours)}")

        # Save only the composite image showing all steps
        base_filename = os.path.join(final_dir, f'image_{i}_{prediction}.jpg')
        cv2.imwrite(base_filename, segmented_image)
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes[0, 0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('Original Image')
        axes[0, 1].imshow(negative_transformation(original_image), cmap='gray')
        axes[0, 1].set_title('Negative Image')
        axes[0, 2].imshow(processed_image, cmap='gray')
        axes[0, 2].set_title('Contrast Stretched Image')
        axes[1, 0].imshow(edged, cmap='gray')
        axes[1, 0].set_title('Edged Image')
        axes[1, 1].imshow(mask, cmap='gray')
        axes[1, 1].set_title('Mask')
        axes[1, 2].imshow(cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB))
        axes[1, 2].set_title(f'Segmented Object - {prediction}')
        # plt.savefig(base_filename)
        plt.close(fig)

        from sklearn.metrics import precision_score, recall_score, f1_score

# Compute precision, recall, and F1-score for each label
precision = precision_score(test_labels, predictions, average=None, labels=unique_labels)
recall = recall_score(test_labels, predictions, average=None, labels=unique_labels)
f1 = f1_score(test_labels, predictions, average=None, labels=unique_labels)

# Print precision, recall, and F1-score for each label
print('Precision for each label:', precision)
print('Recall for each label:', recall)
print('F1-score for each label:', f1)

# Compute Dice coefficient for each label
dice_coefficients = 2 * (precision * recall) / (precision + recall)

# Print Dice coefficient for each label
print('Dice Coefficient for each label:', dice_coefficients)

# Plot Dice Coefficient and F1-score using seaborn
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.barplot(x=unique_labels, y=dice_coefficients, palette='Blues')
plt.title('Dice Coefficient for each label')
plt.xlabel('Label')
plt.ylabel('Dice Coefficient')

plt.subplot(1, 2, 2)
sns.barplot(x=unique_labels, y=f1, palette='Greens')
plt.title('F1-score for each label')
plt.xlabel('Label')
plt.ylabel('F1-score')

plt.tight_layout()
plt.show()


