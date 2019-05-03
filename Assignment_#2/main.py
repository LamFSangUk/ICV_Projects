import os

import util
import nlm_filter
from evaluation import psnr

import matplotlib.pyplot as plt
import numpy as np
import time

dir_clean = './data/data_clean'
dir_noisy = './data/data_noisy'

var_gauss = [20, 30, 40]
if not os.listdir(dir_noisy):
    util.generate_noisy_image(var_gauss)

count = 0
for filename in os.listdir(dir_noisy):
    img = util.load_image(dir_noisy+'/'+filename)
    img = img[:, :, :3]

    std_deviation = var_gauss[count % 3]

    start = time.time()
    img_d = nlm_filter.nlmeans(img, std_deviation)
    end = time.time()

    min_intensity = img_d.min()
    max_intensity = img_d.max()
    img_d = (img_d - min_intensity) / (max_intensity - min_intensity) * 255
    img_d = img_d.astype(np.uint8)
    plt.imshow(img_d)
    plt.imsave("denoised_"+filename, img_d)

    img_clean = 0
    if filename.startswith('afghan'):
        img_clean = util.load_image(dir_clean+'/'+'afghan_clean.png')
    else:
        img_clean = util.load_image(dir_clean+'/'+'monkey_clean.png')

    psnr_val = psnr(img_clean, img_d)
    print('PSNR: ', psnr_val)
    print('Time: ', end-start)

    count+=1
