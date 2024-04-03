import numpy as np
import matplotlib.pyplot as plt
import cv2


braille_dict = {
    '100000': 'a', '101000': 'b', '110000': 'c', '110100': 'd', '100100': 'e',
    '111000': 'f', '111100': 'g', '101100': 'h', '011000': 'i', '011100': 'j',
    '100010': 'k', '101010': 'l', '110010': 'm', '110110': 'n', '100110': 'o',
    '111010': 'p', '111110': 'q', '101110': 'r', '011010': 's', '011110': 't',
    '100011': 'u', '101011': 'v', '011101': 'w', '110011': 'x', '110111': 'y',
    '100111': 'z', '000000': ' ', '000001': ' ', '000011': ',', '001111': '.'
}


def show_image(img, figsize=(15,15), gray=True):
    plt.figure(figsize=figsize)
    plt.imshow(img, 'gray' if gray else None)
    plt.axis('off')
    
def padding(img, pad_size):
      img = np.pad(img, pad_size, mode='constant', constant_values=0)
      return img
def min_filter(img, size):
    shift = size//2
    img2 = img.copy()
    height, width = np.shape(img)
    for rows in range(shift, height-shift):
        for columns in range(shift,width-shift):
            window = img[rows-shift:rows+shift+1,columns-shift:columns+shift+1]
            img2[rows][columns] = np.min(window)
    return img2
def max_filter(img, size):
    shift = size//2
    img2 = img.copy()
    height, width = np.shape(img)
    for rows in range(shift, height-shift):
        for columns in range(shift,width-shift):
            window = img[rows-shift:rows+shift+1,columns-shift:columns+shift+1]
            img2[rows][columns] = np.max(window)
    return img2
def median_filter(img, size):
    shift = size//2
    img2 = img.copy()
    height, width = np.shape(img)
    for rows in range(shift, height-shift):
        for columns in range(shift,width-shift):
            window = img[rows-shift:rows+shift+1,columns-shift:columns+shift+1]
            img2[rows][columns] = np.median(np.sort(window, axis=None))
    return img2

def create_mask():
    mask = np.ones((7,7))
    mask = mask/(7*7)
    return mask
def apply_filter(img, filter):
    height, width = np.shape(img)
    fheight, fwidth = np.shape(filter)
    half = fheight//2
    new_image = img.copy()
    for rows in range(len(filter)//2, height-len(filter)//2):
        for columns in range(len(filter)//2, width-len(filter)//2):
            sum = 0
            for f_row in range(fheight):
                shifty = half - f_row
                for f_column in range(fwidth):
                    shiftx = half - f_column
                    sum+=filter[f_row][f_column]*img[rows-shifty][columns-shiftx]
            new_image[rows][columns] = sum
    return new_image
def contrast_stretching(image):
    height, width = np.shape(image)
    frequency_array = np.zeros(256)
    for rows in range(height):
        for columns in range(width):
            frequency_array[image[rows][columns]] += 1
    low_percentile = np.percentile(image, 5)
    high_percentile = np.percentile(image, 95)
    image = np.int16(image)
    image[image < low_percentile] = 0
    image[image > high_percentile] = 255
    image = ((image-low_percentile)*255)//(high_percentile-low_percentile)
    image[image < 0] = 0
    image[image > 255] = 255
    image = np.uint8(image)
    return image

def normalize(img):
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX) 
    return img
image = cv2.imread("Original Images/IMD049.bmp",0)
show_image(image)

new_image = padding(image,1)
new_image1 = max_filter(new_image,7)
new_image2 = max_filter(new_image1,3)
show_image(new_image2)
new_image3 = contrast_stretching(new_image2)
show_image(new_image3)
filter = create_mask()
new_image4 = normalize(apply_filter(new_image3,filter))
show_image(new_image4)
a, new_image5 = cv2.threshold(image, 135, 1, cv2.THRESH_BINARY_INV)
show_image(new_image5)
new_image6 = median_filter(new_image5,3)
show_image(new_image6)
new_image7 = min_filter(new_image6,7)
new_image8 = min_filter(new_image7,7)
show_image(new_image8)
lesion_image = cv2.imread("Ground Truths/IMD049_lesion.bmp",0)
FN = lesion_image - new_image8



