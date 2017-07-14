import openslide
import numpy as np
from skimage.color import rgb2hed
from sklearn import preprocessing
from scipy import ndimage, misc

Normal_Path = 'D:\Study\TIF\Train_Normal_Part1'
Tumor_Path = 'D:\Study\TIF\Train_Tumor_Part1'

def colourDeconvolution(ihc_rgb):
    ihc_hed = rgb2hed(ihc_rgb)

    img = ndimage.median_filter(ihc_hed[:, :, 0], size=3)
    minValue = np.min(img)
    img = img - minValue
    return img

def readSlider(filename, x, y, w, h):
    path = '{}\\{}'.format(Normal_Path, filename)
    slide = openslide.OpenSlide(path)

    [m, n] = slide.level_dimensions[0]
    print('m = {}, n = {}'.format(m, n))

    tile = np.array(slide.read_region((x, y), 0, (w, h)))
    slide.close()

    result = tile[:, :, 1:4]
    return result, colourDeconvolution(result)


from skimage import io
def readImage(filename):
    path = './data/{}'.format(filename)
    img = io.imread(path)
    return img, colourDeconvolution(img)



