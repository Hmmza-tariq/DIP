import numpy as np
import cv2 as cv


# task 1
r = 500
pad_width = 8


def create_image(r, padding):
    img = np.ones((r, r))
    return np.pad(img, pad_width=padding, mode='constant', constant_values=0)


matrix = create_image(r, pad_width)
print(matrix.shape)
cv.imshow("image", matrix)
cv.waitKey()


# task 2
# img = cv.imread("img.jpg", 0)
#
# inp = int(input("Enter levels: "))
# x = int(256 / inp)
# for i in range(img.shape[0]):
#     for j in range(img.shape[1]):
#         temp = int(img[i][j] / x)
#         img[i][j] = temp * x
#
# cv.imshow("Output", img)
# cv.waitKey()




# task 3
# r = 500
# pad_width = 4
#
#
# def create_image(r, padding):
#     img = np.ones((r, r, 3), dtype='uint8')
#     w = int(r/8)
#     img[:, :, :] = [255, 255, 255]
#     img[0:w, 0:w, :] = [0, 0, 255]
#     img[500-w:500, 500-w:500, :] = [0, 0, 0]
#     img[500-w:500, 0:w, :] = [255, 0, 0]
#     img[0:w, 500-w:500, :] = [0, 255, 0]
#     return img
#
#
# matrix = create_image(r, pad_width)
# print(matrix.shape)
# cv.imshow("image", matrix)
# cv.waitKey()

# task 4
# image = cv.imread('img.jpg')
# vertical_flip = cv.flip(image, 0)
# horizontal_flip = cv.flip(image, 1)
# both_flip = cv.flip(image, -1)
#
# upper_row = np.hstack((image, vertical_flip))
# lower_row = np.hstack((horizontal_flip, both_flip))
# concatenated_image = np.vstack((upper_row, lower_row))
#
# cv.imshow("Flipped Images", concatenated_image)
# cv.waitKey(0)

# task 5
#
# def create_distance_map(img_size, dist_type):
#     image = np.zeros((img_size, img_size), dtype=np.uint8)
#
#     center = img_size // 2
#
#     for i in range(img_size):
#         for j in range(img_size):
#             if dist_type == 'euclidean':
#                 distance = np.sqrt((i - center) ** 2 + (j - center) ** 2)
#             elif dist_type == 'city_block':
#                 distance = abs(i - center) + abs(j - center)
#             elif dist_type == 'chess_board':
#                 distance = max(i - center, j - center)
#             else:
#                 raise ValueError("Invalid distance type.")
#             distance = np.clip(distance, 0, 255)
#             image[i, j] = distance
#
#     return image
#
#
# size = 501
# inp = int(input("Enter distance type: \n1: city_block\n2: euclidean\n3: chess_board\n"))
# if inp == 1:
#     distance_map = create_distance_map(size, 'city_block')
#     cv.imshow("City Block Distance Map", distance_map)
# elif inp == 2:
#     distance_map = create_distance_map(size, 'euclidean')
#     cv.imshow("Euclidean Distance Map", distance_map)
# elif inp == 3:
#     distance_map = create_distance_map(size, 'chess_board')
#     cv.imshow("Chess Board Distance Map", distance_map)
# else:
#     print("Invalid input. Please choose 1, 2, or 3.")
#
# cv.waitKey(0)
#
# cv.destroyAllWindows()
