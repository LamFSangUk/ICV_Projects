from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io
import matplotlib.pyplot as plt
import numpy as np

from scipy.spatial import Delaunay

class Superpixel:
    def __init__(self):
        self.segment = None
        self.nb_segment = None
        self.centers = None

    def load_image(self, dir):
        return img_as_float(io.imread(dir))

    def convert(self, img, show=True):
        self.segment = slic(img, 750, compactness=30, sigma=1)

        self.nb_segment = len(np.unique(self.segment))

        # centers
        self.centers = np.array([np.mean(np.nonzero(self.segment == i), axis=1) for i in range(self.nb_segment)])

        # show segments
        if show is True:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.imshow(mark_boundaries(img, self.segment))
            plt.axis('off')
            plt.show()

    def get_coordinates_of_pixels(self, idx_superpixel):
        idxes = np.where(self.segment == idx_superpixel)

        list_of_coordinates = list(zip(idxes[0], idxes[1]))

        return list_of_coordinates

    def get_neighbor(self):
        tri = Delaunay(self.centers)
        return tri.vertex_neighbor_vertices

if __name__=="__main__":

    sp=Superpixel()
    image = img_as_float(io.imread('./data/train/8_15_s.bmp'))
    sp.convert(image)
    print(sp.get_neighbor())

