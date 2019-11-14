from skimage import img_as_float
from skimage import io, color, morphology
import matplotlib.pyplot as plt

image = img_as_float(color.rgb2gray(io.imread('line.png')))
image_binary = image < 0.5
out_skeletonize = morphology.skeletonize(image_binary)
out_thin = morphology.thin(image_binary)


f, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(10, 3))

ax0.imshow(image, cmap='gray')
ax0.set_title('Input')

ax1.imshow(out_skeletonize, cmap='gray')
ax1.set_title('Skeletonize')

ax2.imshow(out_thin, cmap='gray')
ax2.set_title('Thin')

plt.savefig('newFIg.jpg')
plt.show()
