from matplotlib.image import imread, imsave
import numpy as np

import os


def load_image(filename):
    img = imread(filename)
    img = (img * 255).astype(np.uint8)

    return img


def add_gaussian_noise(img_clear, std_deviation):
    temp_img = np.float64(np.copy(img_clear))

    height  = img_clear.shape[0]
    width   = img_clear.shape[1]

    img_noisy = np.zeros(img_clear.shape, np.float64)

    if len(img_clear.shape) == 2:   # for grayscale img
        noise = np.random.normal(0, std_deviation, (height, width))
        img_noisy = temp_img + noise
    else:                           # for color img
        noise_r = np.random.normal(0, std_deviation, (height, width))
        noise_g = np.random.normal(0, std_deviation, (height, width))
        noise_b = np.random.normal(0, std_deviation, (height, width))

        img_noisy[:, :, 0] = temp_img[:, :, 0] + noise_r
        img_noisy[:, :, 1] = temp_img[:, :, 1] + noise_g
        img_noisy[:, :, 2] = temp_img[:, :, 2] + noise_b

    return img_noisy


data_clean_dir = './data/data_clean'
data_noisy_dir = './data/data_noisy'


def generate_noisy_image(list_var_gauss):
    if not os.listdir(data_noisy_dir):
        filenames = os.listdir(data_clean_dir)

        for filename in filenames:
            if filename.endswith('.png'):
                img_clear = load_image(data_clean_dir + '/' + filename)

                filename, ext = filename.split('.')
                filename = filename[:-5]

                for var_gauss in list_var_gauss:
                    img_noisy = add_gaussian_noise(img_clear, var_gauss)

                    img_noisy = np.clip(img_noisy, 0, 255)

                    min_intensity = img_noisy.min()
                    max_intensity = img_noisy.max()
                    img_noisy = (img_noisy - min_intensity) / (max_intensity - min_intensity) * 255

                    img_noisy = img_noisy.astype(np.uint8)

                    imsave(data_noisy_dir + '/' + filename + 'noisy_' + str(var_gauss) + '.' + ext, img_noisy)

    return


def craete_gaussian_pyramid(img):
    img_pyramid = [img ]

    kernel = gkern()

    for i in range(2):
        height = img.shape[0]
        width = img.shape[1]

        img_blur = np.zeros(img.shape)
        img_padded = np.pad(img, ((1,1),(1,1),(0,0)), 'symmetric')

        subsample_height = img.shape[0]//2
        subsample_width = img.shape[1]//2

        img_subsampled = np.zeros(shape = (subsample_height,subsample_width,img.shape[2]))

        for y in range(height):
            for x in range(width):
                y_padded = y + 1
                x_padded = x + 1

                patch = img_padded[y_padded-1:y_padded+1+1,x_padded-1:x_padded+1+1,:]
                patch = np.double(patch)

                img_blur[y, x, 0] = (kernel * patch[:, :, 0]).sum()
                img_blur[y, x, 1] = (kernel * patch[:, :, 1]).sum()
                img_blur[y, x, 2] = (kernel * patch[:, :, 2]).sum()

        for y in range(subsample_height):
            for x in range(subsample_width):
                img_subsampled[y, x, :] = img_blur[2*y, 2*x, :]

        # normalize
        min_intensity = img_subsampled.min()
        max_intensity = img_subsampled.max()
        img_subsampled = (img_subsampled - min_intensity) / (max_intensity - min_intensity) * 255
        img_subsampled = np.uint8(img_subsampled)

        img_pyramid.append(img_subsampled)
        img = img_subsampled

    return img_pyramid




def gkern(l=3, sig=1.):
    """
    creates gaussian kernel with side length l and a sigma of sig
    """

    ax = np.arange(-l // 2 + 1., l // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)

    kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sig))

    return kernel / np.sum(kernel)

if __name__ == '__main__':
    # img = load_image('afghan_clean.png')
    #
    # img_noisy = add_gaussian_noise(img, 20)
    #
    # min = img_noisy.min()
    # max = img_noisy.max()
    # img_noisy = (img_noisy - min) / (max - min) * 255
    #
    # plt.imshow(img_noisy.astype(np.uint8))
    # plt.show()
    generate_noisy_image([20, 30, 40])
    #img = load_image('./data/data_clean/afghan_clean.png')
    #py= craete_gaussian_pyramid(img)
    #import matplotlib.pyplot as plt
    #plt.imshow(py[2])
    #plt.show()
