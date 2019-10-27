import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
from skimage.data import binary_blobs
import cv2
import numpy as np
inputpath="D:/PROJECTS/Python/car video/found7/crop/number/digits/"
outputpath="D:/PROJECTS/Python/car video/found7/crop/"

#data = binary_blobs(200, blob_size_fraction=.2, volume_fraction=.35, seed=1)
data = cv2.imread(inputpath+'-919.jpg-262.jpg-184.jpg-205-13.jpg',0)
# Change gray image to binary
data=np.where(data>np.mean(data),1.0,0.0)


skeleton = skeletonize(data )
skeleton_lee = skeletonize(data, method='lee')

fig, axes = plt.subplots(1, 3, figsize=(8, 4), sharex=True, sharey=True)
ax = axes.ravel()

ax[0].imshow(data, cmap=plt.cm.gray)
ax[0].set_title('original')
ax[0].axis('off')

ax[1].imshow(skeleton, cmap=plt.cm.gray)
ax[1].set_title('skeletonize')
ax[1].axis('off')

ax[2].imshow(skeleton_lee, cmap=plt.cm.gray)
ax[2].set_title('skeletonize (Lee 94)')
ax[2].axis('off')

fig.tight_layout()
plt.show()