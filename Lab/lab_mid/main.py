import cv2
import numpy as np

def braille_to_text(image_path):
    
    braille_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    black_image = cv2.imread("Lab/lab_mid/black.png", cv2.IMREAD_GRAYSCALE)

    braille_dict = {
        '100000': 'a', '101000': 'b', '110000': 'c', '110100': 'd', '100100': 'e',
        '111000': 'f', '111100': 'g', '101100': 'h', '011000': 'i', '011100': 'j',
        '100010': 'k', '101010': 'l', '110010': 'm', '110110': 'n', '100110': 'o',
        '111010': 'p', '111110': 'q', '101110': 'r', '011010': 's', '011110': 't',
        '100011': 'u', '101011': 'v', '011101': 'w', '110011': 'x', '110111': 'y',
        '100111': 'z', '000000': ' ', '000001': ' ', '000011': ',', '001111': '.'
    }


    _, threshold = cv2.threshold(braille_image, 120, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    diameters = [cv2.contourArea(cnt)**0.5 for cnt in contours]
    avg_diameter = sum(diameters) / len(diameters)

    cell_width = int(avg_diameter * 2.5)
    cell_height = int(avg_diameter * 4)
    avg_diameter = int(avg_diameter)
    braille_image_resized = braille_image[avg_diameter:braille_image.shape[0]-avg_diameter, avg_diameter:braille_image.shape[1]-avg_diameter]
    

    braille_image_resized = cv2.resize(braille_image_resized, (cell_width *42, cell_height * 45))
    black_image_resized = cv2.resize(black_image, (avg_diameter, avg_diameter))

    cells = []

    
    for y in range(0, braille_image_resized.shape[0], cell_height):
        for x in range(0, braille_image_resized.shape[1], cell_width):
            cell = braille_image_resized[y:y+cell_height, x:x+cell_width]
            cells.append(cell)

    text = ''
    for cell in cells:
        braille_code = ''
        
        regions = [cell[i*cell.shape[0]//4:(i+1)*cell.shape[0]//4, j*cell.shape[1]//2:(j+1)*cell.shape[1]//2] for i in range(3) for j in range(2)]
        for region in regions:
            _, labels, stats, centroids = cv2.connectedComponentsWithStats(region, connectivity=8)
            if len(stats) > 1:
                braille_code += '1'
            else:
                braille_code += '0'
        
        cv2.imshow(braille_code, cell)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print('Braille Code:', braille_code)
        text += braille_dict.get(braille_code, '?')

    return text


image_path = 'Lab/lab_mid/Braille.png'

english_text = braille_to_text(image_path)

print("English Text:")
print(english_text)



