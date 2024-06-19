import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from skimage.feature import hog
from skimage import color
from sklearn.preprocessing import LabelEncoder

# Define paths
train_path = 'Project\\train'
test_path = 'Project\\test'

# Function to extract HOG features
def extract_hog_features(image):
    gray_image = color.rgb2gray(image)
    hog_features, hog_image = hog(gray_image, pixels_per_cell=(16, 16),
                                  cells_per_block=(2, 2), visualize=True, multichannel=False)
    return hog_features

# Load and preprocess data
def load_data(path):
    data = []
    labels = []
    for category in os.listdir(path):
        category_path = os.path.join(path, category)
        if os.path.isdir(category_path):
            for img_name in os.listdir(category_path):
                img_path = os.path.join(category_path, img_name)
                img = cv2.imread(img_path)
                img = cv2.resize(img, (256, 256))
                features = extract_hog_features(img)
                data.append(features)
                labels.append(category)
    return np.array(data), np.array(labels)

# Load train and test data
X_train, y_train = load_data(train_path)
X_test, y_test = load_data(test_path)

# Encode labels
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)

# Train SVM classifier
svm = SVC(kernel='linear')
svm.fit(X_train, y_train)

# Predict on test data
y_pred = svm.predict(X_test)

# Evaluate classifier
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')

print(f'Classification Accuracy: {accuracy}')
print(f'Confusion Matrix: \n{conf_matrix}')
print(f'F1 Score: {f1}')

# Function to perform basic segmentation using thresholding
def segment_image(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresholded = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)
    return thresholded

# Evaluate segmentation (dummy example, replace with actual mask evaluation)
def evaluate_segmentation(X, y_true_masks):
    dice_scores = []
    for i, image in enumerate(X):
        pred_mask = segment_image(image)
        true_mask = y_true_masks[i]
        intersection = np.logical_and(pred_mask, true_mask)
        dice_score = 2. * intersection.sum() / (pred_mask.sum() + true_mask.sum())
        dice_scores.append(dice_score)
    return np.mean(dice_scores)

# Load segmentation masks for evaluation (dummy example)
# Assuming masks are preprocessed and loaded as numpy arrays
X_seg_train = [cv2.imread(os.path.join(train_path, 'safe', img_name)) for img_name in os.listdir(os.path.join(train_path, 'safe'))]
Y_seg_train = [cv2.imread(os.path.join(train_path, 'annotations', mask_name), 0) for mask_name in os.listdir(os.path.join(train_path, 'annotations'))]

dice_coeff = evaluate_segmentation(X_seg_train, Y_seg_train)
print(f'Dice Coefficient: {dice_coeff}')

# Plot sample outputs
def plot_sample_predictions(images, masks, n_samples=3):
    fig, axes = plt.subplots(n_samples, 2, figsize=(10, 10))
    for i in range(n_samples):
        pred_mask = segment_image(images[i])
        axes[i, 0].imshow(masks[i], cmap='gray')
        axes[i, 0].set_title('Ground Truth Mask')
        axes[i, 1].imshow(pred_mask, cmap='gray')
        axes[i, 1].set_title('Predicted Mask')
    plt.tight_layout()
    plt.show()

plot_sample_predictions(X_seg_train, Y_seg_train)

# Save and submit
# Prepare the submission zip file and GitHub repository manually
