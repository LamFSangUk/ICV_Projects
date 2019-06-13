from skimage import io
import numpy as np

from Inference import Inference

infer = Inference()

import glob
test_file_list = glob.glob('./data/test/*.bmp')
print(test_file_list)

for test_file in test_file_list:
    res = infer.infer(test_file)
    res = res.astype(np.uint8)

    idx_name = test_file.split('\\')[-1]
    idx_name = idx_name.split('s')[0]

    res_dir = './res/inference_new/' + idx_name + 'infer.bmp'

    io.imsave(res_dir, res)