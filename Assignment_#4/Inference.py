import pickle
import numpy as np
import matplotlib.pyplot as plt
from skimage import io

from Superpixel import Superpixel

def chi_square_dist(h1, h2):
    h1, h2 = h1 - h2, h1 + h2
    nonzero = h2 > 0

    h1 = np.square(h1)
    h1[nonzero] = h1[nonzero] / h2[nonzero]
    res = np.nansum(h1) / 2.
    return res

class Inference:
    def __init__(self):
        self.dictionary = None
        self.cluster = None

        dict_dir = './model/voca_model/dictionary_original'
        self.load_dictionary(dict_dir)

        cluster_dir = './model/k_means_model/k_means_cluster_original'
        self.load_cluster(cluster_dir)

        self.color = {  'void':         (0, 0, 0),
                        'building':     (128, 0, 0),
                        'grass':        (0, 128, 0),
                        'tree':         (128, 128, 0),
                        'cow':          (0, 0, 128),
                        'horse':        (128, 0, 128),
                        'sheep':        (0, 128, 128),
                        'sky':          (128, 128, 128),
                        'mountain':     (64, 0, 0),
                        'aeroplane':    (192, 0, 0),
                        'face':         (192, 128, 0),
                        'car':          (64, 0, 128),
                        'bicycle':      (192, 0, 128)}

    def load_dictionary(self, dir):
        with open(dir, "rb") as f:
            self.dictionary = pickle.load(f)

    def load_cluster(self, dir):
        with open(dir, "rb") as f:
            self.cluster = pickle.load(f)

    def infer(self, img_dir):
        sp = Superpixel()
        img = sp.load_image(img_dir)

        # result img
        res_img = np.empty((img.shape[0], img.shape[1], 3))

        sp.convert(img, show=False)

        feature_dir = img_dir.split('\\')[-1]
        idx_name = feature_dir.split('s')[0]
        feature_dir = './data/features_original/' + idx_name + 'feature'
        features = None
        with open(feature_dir, 'rb') as f:
            features = pickle.load(f)

        for i in range(sp.nb_segment):
            coords = sp.get_coordinates_of_pixels(i)

            nn = []
            for coord in coords:
                print(idx_name)
                feature = features[coord[0]][coord[1]]
                feature = feature.reshape(1, -1)
                nn.append(self.cluster.predict(feature))

            hist, bins = np.histogram(nn, bins=np.arange(151), density=True)

            # find category of superpixel
            min_dist = -1
            category = None
            dists = []
            for k, v in self.dictionary.items():
                dist = chi_square_dist(hist, v)
                dists.append(dist)
                if min_dist > dist or min_dist == -1:
                    min_dist = dist
                    category = k

            print(category)
            print(dists)

            # colorize
            for coord in coords:
                res_img[coord[0]][coord[1]] = self.color.get(category, (0, 0, 0))

            # visualize histogram of superpixel
            #plt.bar(np.arange(len(hist)), hist)

            #plt.show()

        return res_img


if __name__=="__main__":
    infer = Inference()

    import glob
    test_file_list = glob.glob('./data/train\\8_14_s.bmp')
    print(test_file_list)

    for test_file in test_file_list:
        res = infer.infer(test_file)
        res = res.astype(np.uint8)

        idx_name = test_file.split('\\')[-1]
        idx_name = idx_name.split('s')[0]

        res_dir = './res/' + idx_name + 'infer.bmp'

        io.imsave(res_dir, res)


