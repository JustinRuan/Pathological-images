import openslide
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from skimage import color,morphology,io,feature,filters, measure,segmentation,util
from skimage.color import rgb2hed
from matplotlib.colors import LinearSegmentedColormap

Normal_Path = 'D:\Study\TIF\Train_Normal_Part1'
Tumor_Path = 'D:\Study\TIF\Train_Tumor_Part1'

def readSlider(filename):
    path = '{}\\{}'.format(Normal_Path, filename)
    slide = openslide.OpenSlide(path)

    [m, n] = slide.level_dimensions[0]
    print('m = {}, n = {}'.format(m, n))

    tile = np.array(slide.read_region((69000, 113000), 0, (1000, 1000)))
    slide.close()

    result = tile[:,:,1:4]
    return result



################################################################

# 读入图像
cmap_hema = LinearSegmentedColormap.from_list('mycmap', ['white', 'navy'])
cmap_dab = LinearSegmentedColormap.from_list('mycmap', ['white',
                                                        'saddlebrown'])
cmap_eosin = LinearSegmentedColormap.from_list('mycmap', ['darkviolet',
                                                          'white'])

# ihc_rgb = io.imread('../data/图像_3950.tif')
ihc_rgb = readSlider('Normal_001.tif')
ihc_hed = rgb2hed(ihc_rgb)

## 预处理开始
# HaeImage = ihc_hed[:, :, 0]
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
HaeImage = min_max_scaler.fit_transform(ihc_hed[:, :, 0])

# import cv2
# temp1 = (255 * HaeImage).astype(np.ubyte)
# HaeImage = cv2.medianBlur(temp1, 5)

hist = np.histogram(HaeImage, bins=100, normed=True)

Thresholding = filters.threshold_otsu(HaeImage)
ThreshImage = (HaeImage > Thresholding)

tempImage = morphology.remove_small_objects(ThreshImage, min_size=300,connectivity=1)

from scipy import ndimage as ndi
img_cleaned = ndi.binary_fill_holes(tempImage)

distance = ndi.distance_transform_edt(img_cleaned)
local_maxi = feature.peak_local_max(distance, min_distance=20,indices=False, labels=img_cleaned)

markers = morphology.label(local_maxi, background=0, return_num=False, connectivity=2)
markers[~img_cleaned] = -1
labels_rw = segmentation.random_walker(distance, markers)#会出错

image_label_overlay = color.label2rgb(labels_rw)

### 特征提取
METHOD = 'uniform'
lbp = np.zeros(HaeImage.shape)
for radius in range(1,6):
    n_points = 8 * radius
    lbp = lbp + feature.local_binary_pattern(HaeImage, n_points, radius, METHOD)

temp = min_max_scaler.fit_transform(lbp)

hist2 = np.histogram(temp, bins=50, normed=False)

select = temp > 0.7
temp[select] = 0


filterImg = filters.rank.median(temp, morphology.disk(5))


###绘图输出
fig, axes = plt.subplots(2, 4, figsize=(8, 3))
ax = axes.ravel()

ax[0].imshow(ihc_rgb)
ax[0].set_title("Original image")

ax[1].imshow(HaeImage, cmap=cmap_hema)
ax[1].set_title("Hematoxylin")

ax[2].imshow(ihc_hed[:, :, 1], cmap=cmap_eosin)
ax[2].set_title("Eosin")

ax[3].imshow(ihc_hed[:, :, 2], cmap=cmap_dab)
ax[3].set_title("DAB")

ax[4].plot(hist[1][:-1], hist[0], lw=2)
ax[4].set_title('histogram of Hematoxylin')

ax[5].imshow(ThreshImage,cmap=plt.cm.gray, interpolation='nearest')
ax[5].set_title("ThreshImage = {:.4f}".format(Thresholding))

ax[6].imshow(img_cleaned,cmap=plt.cm.gray, interpolation='nearest')
ax[6].set_title("img_cleaned")

ax[7].imshow(image_label_overlay, cmap=plt.cm.Spectral)
ax[7].set_title("count ={}".format(np.amax(markers)))

for a in ax.ravel():
    a.axis('off')

ax[4].axis('on')

# fig.tight_layout()

### 第二张图
fig2 = plt.figure(1)
plt.figure(1)
fig2, axes2 = plt.subplots(2, 3, figsize=(8, 3))
ax = axes2.ravel()


# min_max_scaler = preprocessing.MinMaxScaler()
# HaeImage2 = min_max_scaler.fit_transform(HaeImage)

ax[0].imshow(ihc_rgb)
ax[0].contour(labels_rw, [0.5], linewidths=0.5, colors='r')
ax[0].set_title("count ={}".format(np.amax(markers)))

ax[1].imshow(lbp, cmap=plt.cm.gray)
ax[1].set_title('LBP')

ax[2].plot(hist2[1][:-1], hist2[0], lw=2)

ax[3].imshow(filterImg, cmap=plt.cm.gray)
ax[3].set_title('median')

for a in ax.ravel():
    a.axis('off')
ax[2].axis('on')

# fig2.tight_layout()
plt.show()