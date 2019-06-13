import glob
from skimage import io
import numpy as np

class_list = np.array(['void', 'building', 'grass', 'tree', 'cow','horse', 'sheep', 'sky', 'mountain','aeroplane', 'face', 'car', 'bicycle'])
count_list = np.zeros(class_list.shape)
confusion_matrix = np.zeros((class_list.shape[0], class_list.shape[0]), dtype=np.ulonglong)
print(confusion_matrix[12][12])
dict_idx = {  'void':         0,
                        'building':     1,
                        'grass':        2,
                        'tree':         3,
                        'cow':          4,
                        'horse':        5,
                        'sheep':        6,
                        'sky':          7,
                        'mountain':     8,
                        'aeroplane':    9,
                        'face':         10,
                        'car':          11,
                        'bicycle':      12}

dict_category = {(0, 0, 0):     'void',
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

origin_dir = './data/mask'
inference_dir = './res/inference'

inference_file_list = glob.glob(inference_dir+"/*")
nb_total_pixel = 0

for infer_file in inference_file_list:
    idx_name = infer_file.split("\\")[-1]
    idx_name = idx_name.split("infer")[0]

    mask_file = './data/mask/'+idx_name+'s_GT.bmp'

    infer_img = io.imread(infer_file)
    infer_img = infer_img.reshape((-1, 3))
    mask_img = io.imread(mask_file)
    mask_img = mask_img.reshape((-1, 3))

    nb_pixel = infer_img.shape[0]
    nb_total_pixel += nb_pixel
    for i in range(nb_pixel):
        infer_category = dict_category.get(tuple(infer_img[i]))
        mask_category = dict_category.get(tuple(mask_img[i]))

        infer_idx = dict_idx.get(infer_category)
        mask_idx = dict_idx.get(mask_category, 0)
        count_list[mask_idx] += 1

        confusion_matrix[mask_idx][infer_idx] += 1

    print(idx_name)
    print(count_list)
confusion_matrix_percent = np.zeros((class_list.shape[0], class_list.shape[0]), dtype=np.longdouble)
confusion_matrix_percent = confusion_matrix / np.double(count_list) * 100
with np.printoptions(precision=3, suppress=True):
    print(confusion_matrix_percent)