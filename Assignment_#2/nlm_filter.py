import numpy as np
import util

def nlmeans(img, var_gauss):

    # For color image
    if 0.0 < var_gauss <= 25.0:
        return _nlmeans(img, var_gauss, 10, 1, 0.55)
    elif 25.0 < var_gauss <= 55.0:
        return _nlmeans(img, var_gauss, 17, 2, 0.4)
    elif var_gauss <= 100.0:
        return _nlmeans(img, var_gauss, 17, 3, 0.35)
    else:
        raise ValueError('var_gauss must be between 0 to 100')


def _nlmeans(img, var_gauss, rsize, fsize, filter_param):
    height = img.shape[0]
    width = img.shape[1]

    area_patch = 3 * (2 * fsize + 1) * (2 * fsize + 1)

    # result img which will be denoised
    img_denoise = np.empty(img.shape)
    print(img_denoise.dtype)

    # padding boundary
    img = np.pad(img, ((fsize, fsize), (fsize, fsize), (0, 0)), 'symmetric')
    print(img.shape)

    h = filter_param * var_gauss
    h = h**2

    print(img_denoise.shape)

    for y in range(height):
        for x in range(width):
            x_padded = x + fsize
            y_padded = y + fsize

            patch_1 = img[y_padded - fsize:y_padded + fsize + 1, x_padded - fsize:x_padded + fsize + 1, :]
            patch_1 = patch_1.astype(np.float)

            ry_min = max(y_padded - rsize, fsize)
            rx_min = max(x_padded - rsize, fsize)
            ry_max = min(y_padded + rsize, height - 1 - fsize)
            rx_max = min(x_padded + rsize, width - 1 - fsize)

            fsum_weight = 0.0
            fmax_weight = 0.0
            fsum_pixel = np.zeros(3)

            for i in range(ry_min, ry_max + 1):
                for j in range(rx_min, rx_max + 1):
                    if i==y_padded and j==x_padded:
                        continue

                    patch_2 = img[i - fsize:i + fsize + 1, j - fsize:j + fsize + 1, :]
                    patch_2 = patch_2.astype(np.float)

                    d = squared_euclidean_distance(patch_1, patch_2, area_patch)

                    fweight = weight(d, var_gauss, h)

                    if fweight > fmax_weight:
                        fmax_weight = fweight

                    fsum_weight += fweight
                    fsum_pixel += (img[i, j, :] * fweight)

            fsum_weight += fmax_weight
            fsum_pixel += (img[y_padded, x_padded, :] * fmax_weight)

            if fsum_weight > 0:
                img_denoise[y,x,:] = fsum_pixel / fsum_weight
            else:
                img_denoise[y,x,:] = img[y][x]

        print('y=', y)

    return img_denoise


def _nlmeans_modified(img, var_gauss, rsize, fsize, filter_param):
    img_pyramid = util.craete_gaussian_pyramid(img)

    height = img.shape[0]
    width = img.shape[1]
    half_height = img_pyramid[1].shape[0]
    half_width = img_pyramid[1].shape[1]
    quarter_height = img_pyramid[2].shape[0]
    quarter_width = img_pyramid[2].shape[1]


    area_patch = 3 * (2 * fsize + 1) * (2 * fsize + 1)

    # result img which will be denoised
    img_denoise = np.empty(img.shape)
    print(img_denoise.dtype)

    # padding boundary
    img = np.pad(img, ((fsize, fsize), (fsize, fsize), (0, 0)), 'symmetric')
    print(img.shape)

    h = filter_param * var_gauss
    h = h ** 2

    print(img_denoise.shape)

    for y in range(height):
        for x in range(width):
            x_padded = x + fsize
            y_padded = y + fsize

            patch_1 = img[y_padded - fsize:y_padded + fsize + 1, x_padded - fsize:x_padded + fsize + 1, :]
            patch_1 = patch_1.astype(np.float)

            ry_min = max(y_padded - rsize, fsize)
            rx_min = max(x_padded - rsize, fsize)
            ry_max = min(y_padded + rsize, height - 1 - fsize)
            rx_max = min(x_padded + rsize, width - 1 - fsize)

            fsum_weight = 0.0
            fmax_weight = 0.0
            fsum_pixel = np.zeros(3)

            # original image search space
            for i in range(ry_min, ry_max + 1):
                for j in range(rx_min, rx_max + 1):
                    if i == y_padded and j == x_padded:
                        continue

                    patch_2 = img[i - fsize:i + fsize + 1, j - fsize:j + fsize + 1, :]
                    patch_2 = patch_2.astype(np.float)

                    d = squared_euclidean_distance(patch_1, patch_2, area_patch)

                    fweight = weight(d, var_gauss, h)

                    if fweight > fmax_weight:
                        fmax_weight = fweight

                    fsum_weight += fweight
                    fsum_pixel += (img[i][j] * fweight)

            # half-scale image search space
            ry_min = max((y_padded-fsize)//2 + fsize - rsize, fsize)
            rx_min = max((x_padded-fsize)//2 + fsize - rsize, fsize)
            ry_max = min((y_padded-fsize)//2 + fsize + rsize, half_height - 1 - fsize)
            rx_max = min((x_padded-fsize)//2 + fsize + rsize, half_width - 1 - fsize)

            for i in range(ry_min, ry_max + 1):
                for j in range(rx_min, rx_max + 1):
                    if i == y_padded and j == x_padded:
                        continue

                    patch_2 = img_pyramid[1][i - fsize:i + fsize + 1, j - fsize:j + fsize + 1, :]
                    patch_2 = patch_2.astype(np.float)

                    d = squared_euclidean_distance(patch_1, patch_2, area_patch)

                    fweight = weight(d, var_gauss, h)

                    if fweight > fmax_weight:
                        fmax_weight = fweight

                    fsum_weight += fweight
                    fsum_pixel += (img_pyramid[1][i][j] * fweight)

            # quarter-scale image search space
            ry_min = max((y_padded-fsize)//4 + fsize - rsize, fsize)
            rx_min = max((x_padded-fsize)//4 + fsize - rsize, fsize)
            ry_max = min((y_padded-fsize)//4 + fsize + rsize, quarter_height - 1 - fsize)
            rx_max = min((x_padded-fsize)//4 + fsize + rsize, quarter_width - 1 - fsize)

            for i in range(ry_min, ry_max + 1):
                for j in range(rx_min, rx_max + 1):
                    if i == y_padded and j == x_padded:
                        continue

                    patch_2 = img_pyramid[2][i - fsize:i + fsize + 1, j - fsize:j + fsize + 1, :]
                    patch_2 = patch_2.astype(np.float)

                    d = squared_euclidean_distance(patch_1, patch_2, area_patch)

                    fweight = weight(d, var_gauss, h)

                    if fweight > fmax_weight:
                        fmax_weight = fweight

                    fsum_weight += fweight
                    fsum_pixel += (img_pyramid[2][i][j] * fweight)


            fsum_weight += fmax_weight
            fsum_pixel += (img[y_padded][x_padded] * fmax_weight)

            img_denoise[y][x] = fsum_pixel / fsum_weight

        # print('y=', y)

    return img_denoise


def weight(d, sigma, h):
    squared_sigma = sigma**2
    d = np.clip(d-2*squared_sigma, 0.0, None)
    w = np.exp(-d / h)
    return w


def squared_euclidean_distance(patch_1, patch_2, area_patch):
    # patch will be h x w x 3

    patch_diff = patch_1 - patch_2
    #print(patch_diff)
    squared_patch_diff = np.power(patch_diff, 2)
    #print('dff',squared_patch_diff)
    squared_distance = squared_patch_diff.sum() / area_patch

    return squared_distance
