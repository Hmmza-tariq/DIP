import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

#task 1
# img = Image.open('Lab/images/lab_4_1.tif')
# img_array = np.array(img)
# p5, p95 = np.percentile(img_array, (5, 95))
# img_rescale = np.clip(img_array, p5, p95)
# img_rescale = ((img_rescale - p5) / (p95 - p5)) * 255
# img_rescale = Image.fromarray(img_rescale.astype('uint8'))
# img_rescale.save('Lab/images/output_lab_5_1.jpg')



#task 2
img = Image.open('Lab/images/lab_4_1.tif')

img_array = np.array(img)
histogram = np.zeros(256)
for pixel in img_array.flatten():
    histogram[pixel] += 1

plt.figure()
plt.plot(histogram)
plt.savefig('Lab/images/Figure_1.jpg')
pdf = histogram / (img_array.shape[0] * img_array.shape[1])
plt.figure()
plt.plot(pdf)
plt.savefig('Lab/images/Figure_2.jpg')
cdf = pdf.cumsum()
plt.figure()
plt.plot(cdf)
plt.savefig('Lab/images/Figure_3.jpg')
trans_func = cdf * 255
plt.figure()
plt.plot(trans_func)
plt.savefig('Lab/images/Figure_4.jpg')
enhanced_img_array = trans_func[img_array.astype(int)]
enhanced_img = Image.fromarray(enhanced_img_array.astype('uint8'))
enhanced_img.save('Lab/images/Figure_5.jpg')
enhanced_histogram = np.zeros(256)
for pixel in enhanced_img_array.flatten():
    enhanced_histogram[int(pixel)] += 1
plt.figure()
plt.plot(enhanced_histogram)
plt.savefig('Lab/images/Figure_6.jpg')

plt.show()