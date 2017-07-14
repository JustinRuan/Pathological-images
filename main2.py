import numpy as np
import matplotlib.pyplot as plt
from read import readSlider, readImage
from core import FastRadialSymmetryTransform
from sklearn import preprocessing
from skimage import color,morphology,io,feature,filters, measure,segmentation,util
from skimage.morphology import extrema

rawImage = []
HaeImage = []
if True:
   rawImage, HaeImage = readImage('3950.tif')
else:
  rawImage, HaeImage = readSlider('Normal_001.tif', 69000, 113000, 1000, 1000)


frst = FastRadialSymmetryTransform()
ns = np.arange(2,9,2)
# ns = [12]
frstImg = frst.transform(HaeImage, ns, 1, 2)
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
frstImg = min_max_scaler.fit_transform(frstImg)

local_maxi = feature.peak_local_max(frstImg, min_distance=10,indices=False)
local_mimg = morphology.dilation(local_maxi, morphology.disk(6))

hist = np.histogram(HaeImage, bins=100, normed=True)
Thresholding = filters.threshold_otsu(HaeImage)
ThreshImage = (HaeImage > Thresholding)
ThreshImage = morphology.remove_small_objects(ThreshImage, min_size=200,connectivity=1)















###绘图输出
fig, axes = plt.subplots(2, 3, figsize=(5, 3))
ax = axes.ravel()

ax[0].imshow(rawImage)
ax[0].set_title("Original image")

ax[1].imshow(HaeImage, cmap=plt.cm.gray)
ax[1].set_title("Hae image")

ax[2].imshow(frstImg, cmap=plt.cm.gray)
ax[2].set_title("frst image")

temp = rawImage
temp[local_mimg] = [255,0,0]
ax[3].imshow(temp)
ax[3].set_title("local max image")


ax[4].imshow(ThreshImage, cmap=plt.cm.gray)
ax[4].set_title("Thresh image = {:.4f}".format(Thresholding))
# ax[3].set_title("Thresh image")

for a in ax.ravel():
    a.axis('off')
fig.tight_layout()
plt.show()