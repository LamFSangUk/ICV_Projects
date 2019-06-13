from FilterBank import FilterBank
from VisualDict import VisualDict

import sys
import numpy

fb = FilterBank()
vs = VisualDict(fb, "./data/train")

#vs.save_features('./data/image', './data/features_original')
#vs.train(save=True, save_dir="./model/k_means_model/k_means_cluster_original")

#vs.load_cluster('./model/k_means_model/k_means_cluster_original')
#with open('kmeans.txt','w') as f:
#    numpy.set_printoptions(threshold=sys.maxsize)
#    f.write(str(vs.cluster.cluster_centers_))
#vs.create_dictionary(save=True, save_dir='./model/voca_model/dictionary_original')

vs.load_dictionary('./model/voca_model/dictionary_original')
vs.show_dictionary()