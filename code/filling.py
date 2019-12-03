from skimage import io, morphology, img_as_bool, segmentation
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
import cv2

image = img_as_bool(io.imread('gaps.jpg'))
#image = cv2.copyMakeBorder(image1, 10, 10, 10, 10, cv2.BORDER_CONSTANT)
out = ndi.distance_transform_edt(~image)
out = out < 0.05 * out.max()
out = morphology.skeletonize(out)
#out = morphology.binary_dilation(out, morphology.selem.disk(1))
out = segmentation.clear_border(out)
out = out | image

plt.imshow(out, cmap='gray')
plt.imsave('gaps_filled.jpg', out, cmap='gray')
plt.show()


img = cv2.imread('gaps_filled.jpg')
cv2.imshow("img",img)
cv2.imshow("out",out)
cv2.waitKey(0)
