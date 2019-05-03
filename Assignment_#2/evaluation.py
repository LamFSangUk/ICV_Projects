import numpy as np


def mse(image_origin, image_denoised):
    height = image_origin.shape[0]
    width = image_origin.shape[1]

    diff = image_origin - image_denoised
    diff = np.power(diff, 2)
    sum_euclidean_diff = diff.sum()
    mse_val = sum_euclidean_diff / (3*height*width)

    return mse_val


def psnr(image_origin, image_denoised):
    mse_val = mse(image_origin, image_denoised)

    psnr_val = 20 * np.log10(255 / np.sqrt(mse_val))

    return psnr_val


if __name__ == '__main__':
    from util import load_image
    img_clean = load_image('./data/data_clean/afghan_clean.png')
    img_clean = img_clean[:,:,:3]
    img_denoised = load_image('denoise[20, 30, 40]afghan_noisy_30.png')
    img_denoised = img_denoised[:, :, :3]

    print(psnr(img_clean,img_denoised))
