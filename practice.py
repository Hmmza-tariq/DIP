import cv2
import os
import numpy as np
from skimage.color import rgb2gray
from skimage.feature import local_binary_pattern
from skimage import measure
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def load_images_and_masks(data_dir):
    images = []
    labels = []
    categories = ['safe', 'gun', 'knife']
    for category in categories:
        category_path = os.path.join(data_dir, category)
        for file in os.listdir(category_path):
            img_path = os.path.join(category_path, file)
            image = cv2.imread(img_path)
            if image is not None:
                images.append(image)
                labels.append(category)
    return images, labels

def extract_features(images):
    features = []
    for image in images:
        if image.ndim == 3:
            image = rgb2gray(image)
        resized_image = cv2.resize(image, (128, 128))
        
        # Convert image to uint8
        resized_image_uint8 = (resized_image * 255).astype(np.uint8)
        
        # Edge features
        edges = cv2.Canny(resized_image_uint8, 100, 200)
        edge_hist = np.histogram(edges, bins=256, range=(0, 256))[0]
        
        # LBP features
        lbp = local_binary_pattern(resized_image_uint8, P=8, R=1, method='uniform')
        lbp_hist, _ = np.histogram(lbp, bins=np.arange(0, 10), range=(0, 9))
        
        # Hu moments
        moments = cv2.moments(resized_image)
        hu_moments = cv2.HuMoments(moments).flatten()
        
        # HOG features
        fd = hog(resized_image, orientations=9, pixels_per_cell=(8, 8),
                    cells_per_block=(2, 2), visualize=False, feature_vector=True)
        
        # Combine features
        feature_vector = np.hstack((edge_hist, lbp_hist, hu_moments, fd))
        features.append(feature_vector)
        
    return features

def train_svm_classifier(features, labels):
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    clf = SVC(kernel='linear', probability=True)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print('Classification Report:')
    print(classification_report(y_test, y_pred))
    print('Confusion Matrix:')
    print(confusion_matrix(y_test, y_pred))
    print('Accuracy:', accuracy_score(y_test, y_pred))
    return clf, X_test, y_test, y_pred

def segment_image(image, clf):
    if image.ndim == 3:
        image = rgb2gray(image)
    resized_image = cv2.resize(image, (128, 128))
    
    # Convert image to uint8
    resized_image_uint8 = (resized_image * 255).astype(np.uint8)
    
    # Edge features
    edges = cv2.Canny(resized_image_uint8, 100, 200)
    edge_hist = np.histogram(edges, bins=256, range=(0, 256))[0]
    
    # LBP features
    lbp = local_binary_pattern(resized_image_uint8, P=8, R=1, method='uniform')
    lbp_hist, _ = np.histogram(lbp, bins=np.arange(0, 10), range=(0, 9))
    
    # Hu moments
    moments = cv2.moments(resized_image)
    hu_moments = cv2.HuMoments(moments).flatten()
    
    # HOG features
    fd = hog(resized_image, orientations=9, pixels_per_cell=(8, 8),
                    cells_per_block=(2, 2), visualize=False, feature_vector=True)
    
    # Combine features
    feature_vector = np.hstack((edge_hist, lbp_hist, hu_moments, fd)).reshape(1, -1)
    
    prediction = clf.predict(feature_vector)
    if prediction[0] != 'safe':
        ret, thresh = cv2.threshold((resized_image * 255).astype(np.uint8), 128, 255, cv2.THRESH_BINARY)
        labels = measure.label(thresh)
        output_image = cv2.cvtColor((resized_image * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        for region in measure.regionprops(labels):
            minr, minc, maxr, maxc = region.bbox
            cv2.rectangle(output_image, (minc, minr), (maxc, maxr), (0, 255, 0), 2)
            cv2.putText(output_image, prediction[0], (minc, minr - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return output_image, prediction[0]
    return cv2.cvtColor((resized_image * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR), 'safe'

def plot_metrics(labels, predictions, output_dir):
    f1 = f1_score(labels, predictions, average=None, labels=['gun', 'knife', 'safe'])
    dice_coefficient = 2 * f1 / (1 + f1)
    
    plt.figure(figsize=(10, 5))
    plt.bar(['gun', 'knife', 'safe'], f1, color='green')
    plt.title('F1-score for each label')
    plt.xlabel('Label')
    plt.ylabel('F1-score')
    plt.savefig(os.path.join(output_dir, 'f1_scores.png'))
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.bar(['gun', 'knife', 'safe'], dice_coefficient, color='blue')
    plt.title('Dice Coefficient for each label')
    plt.xlabel('Label')
    plt.ylabel('Dice Coefficient')
    plt.savefig(os.path.join(output_dir, 'dice_coefficients.png'))
    plt.close()

def save_confusion_matrix(y_test, y_pred, output_dir):
    conf_matrix = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    cax = ax.matshow(conf_matrix, cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    fig.colorbar(cax)
    labels = ['safe', 'gun', 'knife']
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    
    # Display the numbers inside the matrix blocks
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, conf_matrix[i, j], ha='center', va='center', color='black')

    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()

def plot_classification_report(y_test, y_pred, output_dir):
    report = classification_report(y_test, y_pred, output_dict=True)
    
    labels = list(report.keys())[:-3]  # Exclude 'accuracy', 'macro avg', and 'weighted avg'
    precision = [report[label]['precision'] for label in labels]
    recall = [report[label]['recall'] for label in labels]
    f1_score = [report[label]['f1-score'] for label in labels]
    support = [report[label]['support'] for label in labels]
    
    x = np.arange(len(labels))
    width = 0.2
    
    fig, ax1 = plt.subplots(figsize=(10, 5))
    
    ax1.bar(x - width, precision, width, label='Precision')
    ax1.bar(x, recall, width, label='Recall')
    ax1.bar(x + width, f1_score, width, label='F1-Score')
    
    ax1.set_xlabel('Labels')
    ax1.set_ylabel('Scores')
    ax1.set_title('Classification Report')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.legend()
    
    ax2 = ax1.twinx()
    ax2.plot(x, support, color='black', marker='o', linestyle='-', linewidth=2, label='Support')
    ax2.set_ylabel('Support')
    ax2.legend(loc='upper right')
    
    plt.savefig(os.path.join(output_dir, 'classification_report.png'))
    plt.close()

def generate_segmentation_images(images, labels, classifier, output_dir):
    for idx, (image, label) in enumerate(zip(images, labels)):
        segmented_image, prediction = segment_image(image, classifier)
        
        # Resize the segmented image to match the original image's dimensions
        segmented_image_resized = cv2.resize(segmented_image, (image.shape[1], image.shape[0]))
        
        # Create grid image
        grid_image = np.zeros((image.shape[0], image.shape[1] * 2, 3), dtype=np.uint8)
        grid_image[:, :image.shape[1]] = image
        grid_image[:, image.shape[1]:] = segmented_image_resized

        cv2.putText(grid_image, f'Prediction: {prediction}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), 1)
        cv2.putText(grid_image, f'Actual: {label}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Save the grid image
        plt.figure(figsize=(10, 5))
        plt.grid(False)
        plt.imshow(cv2.cvtColor(grid_image, cv2.COLOR_BGR2RGB))
        plt.title(f'Segment Image {idx}: {label} ({prediction})')
        plt.savefig(os.path.join(output_dir, f'segmented_display_{label}_{idx}.png'))
        plt.close()

def display(paths):
    images = [cv2.imread(path) for path in paths]
    plt.figure(figsize=(16, 4))
    for i in range(len(images)):
        plt.subplot(1, len(images), i+1)
        plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB), aspect='equal')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def main():
    base_dir = 'Project'  
    train_dir = os.path.join(base_dir, 'train')
    test_dir = os.path.join(base_dir, 'test')
    analytics_output_dir = os.path.join(base_dir, 'output/details')
    output_dir = os.path.join(base_dir, 'output')
    os.makedirs(analytics_output_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    train_images, train_labels = load_images_and_masks(train_dir)
    test_images, test_labels = load_images_and_masks(test_dir)

    train_features = extract_features(train_images)
    
    clf, X_test, y_test, y_pred = train_svm_classifier(train_features, train_labels)
    plot_metrics(y_test, y_pred, analytics_output_dir)
    save_confusion_matrix(y_test, y_pred, analytics_output_dir)
    plot_classification_report(y_test, y_pred, analytics_output_dir)
    generate_segmentation_images(test_images, test_labels, clf, output_dir)
    # display([os.path.join(analytics_output_dir, 'confusion_matrix.png'), 
    #          os.path.join(analytics_output_dir, 'dice_coefficients.png'), 
    #          os.path.join(analytics_output_dir, 'f1_scores.png'),
    #          os.path.join(analytics_output_dir, 'classification_report.png')])
    print("Analysis and image generation completed. Check the output folder.")

if __name__ == "__main__":
    main()
