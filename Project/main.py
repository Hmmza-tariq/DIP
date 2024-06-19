import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA

# Paths to the directories
base_dir = 'D:/Semester 6/DIP/DIP-code/Project'
train_dir = os.path.join(base_dir, 'train')
annotations_dir = os.path.join(train_dir, 'annotations')

# Image size for resizing
IMAGE_SIZE = (128, 128)

# Function to load images and labels for classification
def load_data():
    object_classes = ['gun', 'knife', 'safe']  # Add 'shuriken' if applicable
    data = []
    labels = []

    for object_class in object_classes:
        class_dir = os.path.join(train_dir, object_class)
        files = os.listdir(class_dir)
        for file in files:
            file_path = os.path.join(class_dir, file)
            image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if image is not None:
                image_resized = cv2.resize(image, IMAGE_SIZE)
                data.append(image_resized.flatten())
                labels.append(object_class)

    return np.array(data), np.array(labels)

# Load data and labels
data, labels = load_data()

# Encode labels
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels_encoded, test_size=0.2, random_state=42)

# Reduce dimensionality for faster training (optional)
pca = PCA(n_components=50)  # Adjust the number of components as needed
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Train a support vector classifier
classifier = SVC(kernel='linear', random_state=42)
classifier.fit(X_train_pca, y_train)

# Predict on the test set
y_pred = classifier.predict(X_test_pca)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy * 100:.2f}%")
print("Confusion Matrix:")
print(conf_matrix)

# Function to segment images
def segment_images(annotation_path, train_image_path, output_path):
    annotation = cv2.imread(annotation_path, cv2.IMREAD_GRAYSCALE)
    train_image = cv2.imread(train_image_path)

    if annotation is None:
        raise ValueError(f"Failed to read annotation image: {annotation_path}")
    if train_image is None:
        raise ValueError(f"Failed to read train image: {train_image_path}")

    print(f"Annotation shape: {annotation.shape}, Train image shape: {train_image.shape}")

    # Ensure the dimensions match
    if annotation.shape != train_image.shape[:2]:
        print(f"Resizing train image from {train_image.shape[:2]} to {annotation.shape}")
        train_image = cv2.resize(train_image, (annotation.shape[1], annotation.shape[0]))

    # Create a mask for non-zero pixels
    mask = annotation > 0

    # Segment the image using the mask
    segmented_image = np.zeros_like(train_image)
    segmented_image[mask] = train_image[mask]

    # Save the segmented image
    cv2.imwrite(output_path, segmented_image)

# Iterate through each object class for segmentation
object_classes = ['gun', 'knife', 'safe']  # Add 'shuriken' if applicable

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

# Function to calculate Dice coefficient for segmentation evaluation
def dice_coefficient(mask1, mask2):
    intersection = np.sum(mask1[mask2 == 255])
    return (2. * intersection) / (np.sum(mask1) + np.sum(mask2))

# Example evaluation of segmentation (adjust paths accordingly)
# Note: You need ground truth masks for evaluation
# Replace 'path_to_segmented_image' and 'path_to_ground_truth_mask' with actual paths
segmented_mask = cv2.imread('path_to_segmented_image', cv2.IMREAD_GRAYSCALE)
ground_truth_mask = cv2.imread('path_to_ground_truth_mask', cv2.IMREAD_GRAYSCALE)

dice = dice_coefficient(segmented_mask, ground_truth_mask)
print(f"Dice Coefficient: {dice:.4f}")

# Prepare submission
# Ensure your paths and files are correct and exist
submission_dir = 'D:/Semester 6/DIP/DIP-code/Project/submission'
os.makedirs(submission_dir, exist_ok=True)

# Save the report and code files
report_content = """
Title: Digital Image Processing Project

Abstract:
[Your abstract here]

Introduction:
[Your introduction here]

Methodology:
[Your methodology here, including data preparation, model training, and segmentation approach]

Results:
Classification Accuracy: {accuracy * 100:.2f}%
Confusion Matrix:
{conf_matrix}

Dice Coefficient for Segmentation: {dice:.4f}

Discussion:
[Your discussion here]

Conclusion:
[Your conclusion here]

References:
[Your references here]
""".format(accuracy=accuracy, conf_matrix=conf_matrix, dice=dice)

report_path = os.path.join(submission_dir, 'report.txt')
with open(report_path, 'w') as f:
    f.write(report_content)

# Copy the code files to the submission directory
import shutil
code_files = ['main.py']  # Add other code files if needed
for code_file in code_files:
    shutil.copy(code_file, submission_dir)

print("Submission prepared at:", submission_dir)
