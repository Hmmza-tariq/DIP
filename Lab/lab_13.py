import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, accuracy_score


# task 1
df = pd.read_csv('Lab/files/iris.csv')

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


def min_distance_classifier(X_train, y_train, X_test):
    y_pred = []
    for sample in X_test:
        min_distance = float('inf')
        predicted_class = None
        for i, train_sample in enumerate(X_train):
            distance = np.linalg.norm(sample - train_sample)
            if distance < min_distance:
                min_distance = distance
                predicted_class = y_train[i]
        y_pred.append(predicted_class)
    return y_pred


y_pred = min_distance_classifier(X_train, y_train, X_test)
conf_matrix = confusion_matrix(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
accuracy = accuracy_score(y_test, y_pred)

print("Confusion Matrix:")
print(conf_matrix)
print("Precision:", precision)
print("Accuracy:", accuracy)


plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Greens", xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()

# task 2
def read_inputdata(filename):
    df = pd.read_csv(filename)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    return X, y

def calculate_distance(instance1, instance2):
    return np.linalg.norm(instance1 - instance2)

def find_neighbours(k, X_train, y_train, unseen_sample):
    distances = [(calculate_distance(unseen_sample, sample), label) for sample, label in zip(X_train, y_train)]
    distances.sort()
    return distances[:k]

def get_response(nearest_neighbour_array):
    labels = [label for _, label in nearest_neighbour_array]
    return max(set(labels), key=labels.count)

def confusion_matrix_metrics(y_true, y_pred):
    conf_matrix = confusion_matrix(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')
    return conf_matrix, accuracy, precision

def plot_dataset_with_predictions(X_train, y_train, X_test, y_test, y_pred):
    plt.figure(figsize=(10, 8))
    for label in np.unique(y_train):
        plt.scatter(X_train[y_train == label, 0], X_train[y_train == label, 1], label=f'Class {label}', alpha=0.7)
    for i, label in enumerate(np.unique(y_test)):
        plt.scatter(X_test[y_test == label, 0], X_test[y_test == label, 1], marker=f'${i}$', s=200,
                    label=f'Test Class {label} (True)')
    for i, label in enumerate(np.unique(y_pred)):
        plt.scatter(X_test[y_pred == label, 0], X_test[y_pred == label, 1], marker=f'${i}$', edgecolors='k', s=150,
                    label=f'Test Class {label} (Predicted)')

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Dataset with Test Samples and Predictions')
    plt.legend()
    plt.show()


def shuffle_and_split(X, y, test_size=0.5, random_state=None):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

def kNN_test(X_train, y_train, X_test, y_test, k):
    y_pred = []
    for sample in X_test:
        neighbours = find_neighbours(k, X_train, y_train, sample)
        response = get_response(neighbours)
        y_pred.append(response)
    return y_pred


X, y = read_inputdata('Lab/files/iris.csv')
X_train, X_test, y_train, y_test = shuffle_and_split(X, y, test_size=0.5, random_state=42)

k = 5 
y_pred = kNN_test(X_train, y_train, X_test, y_test, k)
conf_matrix, accuracy, precision = confusion_matrix_metrics(y_test, y_pred)

print("Confusion Matrix:")
print(conf_matrix)
print("Accuracy:", accuracy)
print("Precision:", precision)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Greens", xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()

plot_dataset_with_predictions(X_train, y_train, X_test, y_test, y_pred)