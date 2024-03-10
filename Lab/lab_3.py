import numpy as np
import cv2 as cv

# arr = np.array( [ [ '1', 2, 3 ] , [ 6, 5, 4 ] ] )
# print(arr.astype(np.uint8))

# task 1
# For the image given below (provided with the lab handout), apply the connected
# component labelling and count the total number of white objects. First threshold the
# images and then perform connected component analysis algorithm.


def to_grayscale(image):
    return cv.cvtColor(image, cv.COLOR_BGR2GRAY)


def threshold(image, threshold_value):
    return np.where(image > threshold_value, 255, 0)


def connected_components(image):
    label = 0
    labels = np.zeros_like(image, dtype=int)
    equivalences = {}

    # First pass
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i, j] == 255:
                neighbors = [(i - 1, j), (i, j - 1)]
                neighbors = [labels[x, y] for x, y in neighbors if x >= 0 and y >= 0]
                neighbors = [n for n in neighbors if n > 0]
                if not neighbors:
                    label += 1
                    labels[i, j] = label
                else:
                    labels[i, j] = np.amin(neighbors)
                    for n in neighbors:
                        if n != labels[i, j]:
                            equivalences[n] = labels[i, j]

    # Second pass
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i, j] == 255:
                while labels[i, j] in equivalences:
                    labels[i, j] = equivalences[labels[i, j]]

    return len(np.unique(labels)) - 1


# task 2
def calculate_distance(point_1, point_2, choice):
    if choice == 1:
        return np.sqrt(np.sum((np.array(point_1) - np.array(point_2)) ** 2))
    elif choice == 2:
        return np.sum(np.abs(np.array(point_1) - np.array(point_2)))
    elif choice == 3:
        return np.max(np.abs(np.array(point_1) - np.array(point_2)))
    else:
        return "Invalid choice"


# Load the image
img = cv.imread('Task2.PNG')
gray = to_grayscale(img)
thresh = threshold(gray, 127)
num_objects = connected_components(thresh)

print('The total number of white objects is: ', num_objects)

inp = input("Enter the choice of distance: ")
p1x = int(input("Enter the x-component of first point: "))
p1y = int(input("Enter the y-component of first point: "))
p2x = int(input("Enter the x-component of second point: "))
p2y = int(input("Enter the y-component of second point: "))
print(calculate_distance((p1x, p1y), (p2x, p2y), int(inp)))