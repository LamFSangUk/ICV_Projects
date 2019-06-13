import matplotlib.pyplot as plt
from skimage import io, color
import scipy.ndimage as ndimage
import numpy as np
import glob
import pickle

from sklearn.cluster import MiniBatchKMeans

from FilterBank import FilterBank
from Superpixel import Superpixel

class VisualDict:
    def __init__(self, fb, path_train):
        self.fb = fb
        self.nb_features = 3 * len(fb.range_gauss_filter) + len(fb.range_dog_filter) + len(fb.range_log_filter)

        self.path_train = path_train
        self.cluster = None
        self.nb_clusters = 150

        # BoW
        self.dictionary = {'void':      np.zeros(self.nb_clusters),
                           'building':  np.zeros(self.nb_clusters),
                           'grass':     np.zeros(self.nb_clusters),
                           'tree':      np.zeros(self.nb_clusters),
                           'cow':       np.zeros(self.nb_clusters),
                           'horse':     np.zeros(self.nb_clusters),
                           'sheep':     np.zeros(self.nb_clusters),
                           'sky':       np.zeros(self.nb_clusters),
                           'mountain':  np.zeros(self.nb_clusters),
                           'aeroplane': np.zeros(self.nb_clusters),
                           'face':      np.zeros(self.nb_clusters),
                           'car':       np.zeros(self.nb_clusters),
                           'bicycle':   np.zeros(self.nb_clusters)}

        # category information
        self.category = {(0, 0, 0):     'void',
                         (128, 0, 0):   'building',
                         (0, 128, 0):   'grass',
                         (128, 128, 0): 'tree' ,
                         (0, 0, 128):   'cow',
                         (128, 0, 128): 'horse',
                         (0, 128, 128): 'sheep',
                         (128, 128, 128): 'sky',
                         (64, 0, 0):    'mountain',
                         (192, 0, 0):   'aeroplane',
                         (192, 128, 0): 'face',
                         (64, 0, 128):  'car',
                         (192, 0, 128): 'bicycle'}

    def init_dictionary(self, category):
        for key in category:
            self.dictionary[key] = self.nb_features

    def train(self, save=False, save_dir=None):
        train_img_list = glob.glob(self.path_train+"/*")
        print(train_img_list)

        train_features = []

        for img_file in train_img_list:
            img = io.imread(img_file)
            img = color.rgb2lab(img)
            img_features = self.extract_texton_feature(img, self.fb, self.nb_features)
            train_features.extend(img_features)

        train_features = np.array(train_features)
        print(train_features.shape)

        kmeans_cluster = MiniBatchKMeans(n_clusters=self.nb_clusters, verbose=1, max_iter=300)
        kmeans_cluster.fit(train_features)
        print(kmeans_cluster.cluster_centers_)
        print(kmeans_cluster.cluster_centers_.shape)

        self.cluster = kmeans_cluster

        # save kmeans result
        if save is True:
            with open(save_dir, 'wb') as f:
                pickle.dump(self.cluster, f)

    def load_cluster(self, dir_file):
        with open(dir_file, 'rb') as f:
            self.cluster = pickle.load(f)

    def extract_texton_feature(self, img, fb, nb_features):
        # split CIE lab color image
        L, a, b = np.transpose(img, (-1, 0, 1))

        filter_bank = fb.filter_bank

        features = np.empty((nb_features, img.shape[0], img.shape[1]))

        cnt = 0
        for i in fb.range_gauss_filter:
            features[cnt] = ndimage.convolve(L, filter_bank[i])
            features[cnt+1] = ndimage.convolve(a, filter_bank[i])
            features[cnt+2] = ndimage.convolve(b, filter_bank[i])
            cnt += 3

        for i in fb.range_dog_filter:
            features[cnt] = ndimage.convolve(L, filter_bank[i])
            cnt += 1

        for i in fb.range_log_filter:
            features[cnt] = ndimage.convolve(L, filter_bank[i])
            cnt += 1

        features = np.transpose(features, (1, 2, 0))
        features = features.reshape(-1, nb_features)
        print(features.shape)

        return features

    def save_features(self, src, dst):
        bmp_file_list = glob.glob(src + '/*_s.bmp')

        for bmp_file in bmp_file_list:
            img = io.imread(bmp_file)
            img = color.rgb2lab(img)
            img_features = self.extract_texton_feature(img, self.fb, self.nb_features)
            img_features = img_features.reshape(img.shape[0], img.shape[1], self.nb_features)
            print(img_features.shape)

            filename = bmp_file.split('\\')[-1]
            filename = filename.split('s')[0] + 'feature'

            file_dir = dst + '/' + filename

            with open(file_dir, 'wb') as f:
                pickle.dump(img_features, f)

    def create_superpixel_histogram(self):
        sp = Superpixel()

        # Load original image as superpixel
        img_dir = self.path_train
        img_list = glob.glob(img_dir+'\\*.bmp')
        for img_file in img_list:
            img = sp.load_image(img_file)
            sp.convert(img, show=False)

            feature_dir = img_file.split('\\')[-1]
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

                hist, bins = np.histogram(nn, bins=np.arange(self.nb_clusters+1), density=True)

                yield idx_name, coords, hist

    def create_dictionary(self, save=False, save_dir=None):
        import operator

        mask_dir = './data/mask/'
        for idx_name, segment, hist in self.create_superpixel_histogram():
            mask_img = io.imread(mask_dir + idx_name + "s_GT.bmp")

            voting = {}
            for (y, x) in segment:
                rgb = tuple(mask_img[y][x])

                if rgb not in self.category:
                    continue

                category = self.category[rgb]
                voting[category] = voting.get(category, 0) + 1

            if voting:
                print(voting.items())
                max_category = max(voting.items(), key=operator.itemgetter(1))[0]
                self.dictionary[max_category] += hist

        # Normalize
        for k, v in self.dictionary.items():
            if not np.isclose(v.sum(), 0.0):
                self.dictionary[k] /= v.sum()

        print(self.dictionary)

        if save is True:

            with open(save_dir, "wb") as f:
                pickle.dump(self.dictionary, f)

    def load_dictionary(self, dir):
        with open(dir, "rb") as f:
            self.dictionary = pickle.load(f)

    def show_dictionary(self):
        for k, v in self.dictionary.items():
            bins = np.arange(len(v))
            plt.suptitle(k)

            plt.bar(bins, v)
            plt.ylim(ymax=0.5)

            plt.show()


if __name__ == "__main__":
    fb = FilterBank()
    vs = VisualDict(fb, "./data/train")
    #vs.save_features('./data/image', './data/features')
    vs.train(save=True, save_dir="./model/k_means_model/k_means_cluster_200")
    vs.load_cluster('./model/k_means_model/k_means_cluster_200')
    #vs.load_dictionary('./dictionary_with_face')
    vs.create_dictionary(save=True, save_dir='./model/voca_model/dictionary_200')
    #vs.show_dictionary()
