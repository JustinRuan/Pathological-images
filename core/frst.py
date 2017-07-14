import numpy as np
from scipy.ndimage.filters import sobel, gaussian_filter


class FastRadialSymmetryTransform(object):
    def __init__(self, gradient_function=None):
        self._index_cache = {}
        if gradient_function is None:
            gradient_function = self.sobel_gradient
        self.gradient_function = gradient_function

    def index_arrays(self, shape):
        if shape not in self._index_cache:
            self._index_cache[shape] = np.meshgrid(np.arange(shape[1]),
                                                   np.arange(shape[0]))
        return self._index_cache[shape]

    def pixel_map(self, gmag, gx, gy, n):
        x, y = self.index_arrays(gmag.shape)
        gx = gx / gmag
        gy = gy / gmag
        nx = np.rint(gx * n + 0.5).astype(np.int64)
        ny = np.rint(gy * n + 0.5).astype(np.int64)
        posx = x + nx
        posy = y + ny
        negx = x - nx
        negy = y - ny
        return posx, posy, negx, negy

    def transform_component(self, mag, gx, gy, n, sigma, alpha):
        posx, posy, negx, negy = self.pixel_map(mag, gx, gy, n)
        orientation, magnitude = self.orientation_and_magnitude(
            posx, posy, negx, negy, mag)
        o_max = np.max(np.abs(orientation))
        m_max = np.max(np.abs(magnitude))
        orientation = np.abs(orientation) / o_max
        magnitude = np.abs(magnitude) / m_max
        print(o_max, m_max)
        F = self.compute_F(orientation, magnitude, alpha)

        # from scipy.misc import imsave
        # imsave('orientation.jpg', orientation * 255)
        # imsave('magnitude.jpg', magnitude * 255)
        # return F
        return gaussian_filter(F, sigma, mode="constant")

    def transform(self, image, ns, sigmas, alpha):
        transform = np.zeros(image.shape)
        mag, gx, gy = self.gradient_function(image)
        sigmas = np.ones(len(ns)) * sigmas
        insk = zip(range(len(ns)), ns, sigmas)
        for i, n, s in insk:
            transform += self.transform_component(
                mag, gx, gy, n, s, alpha)
        return transform

    def sobel_gradient(self, image):
        grad_y = sobel(image, 0)
        grad_x = sobel(image, 1)
        mag = np.sqrt(grad_y ** 2 + grad_x ** 2) + 1e-16
        return mag, grad_x, grad_y

    def orientation_and_magnitude(self, posx, posy, negx, negy, mag):
        orientation = np.zeros(mag.shape)
        magnitude = np.zeros(mag.shape)
        h, w = mag.shape
        posx[posx < 0] = 0
        posy[posy < 0] = 0
        negx[negx < 0] = 0
        negy[negy < 0] = 0
        posx[posx > w - 1] = w - 1
        posy[posy > h - 1] = h - 1
        negx[negx > w - 1] = w - 1
        negy[negy > h - 1] = h - 1
        np.add.at(orientation, [posy, posx], 1)
        np.add.at(orientation, [negy, negx], -1)
        np.add.at(magnitude, [posy, posx], mag)
        np.add.at(magnitude, [negy, negx], -mag)

        orientation[:,0] = 0
        orientation[:, w - 1] = 0
        orientation[0, :] = 0
        orientation[h - 1, :] = 0
        magnitude[:,0] = 0
        magnitude[:, w - 1] = 0
        magnitude[0, :] = 0
        magnitude[h - 1, :] = 0
        return orientation, magnitude

    def compute_F(self, orientation, magnitude, alpha):
        s = (orientation ** alpha)* magnitude
        s[s < 1e-1] = 0
        return s
