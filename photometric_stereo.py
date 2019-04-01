import numpy as np


def photometric_stereo(img_arr, light_dirs):
    width = img_arr.shape[2]
    height = img_arr.shape[1]

    # light_dir shape : num_imagesX3
    # image_arr : num_imageX(heightXwidth)
    img_arr = img_arr.reshape(img_arr.shape[0], -1)

    #b = np.matmul(np.matmul(np.linalg.inv(np.matmul(light_dirs.T, light_dirs)), light_dirs.T), img_arr)
    b, res, rank, s = np.linalg.lstsq(light_dirs, img_arr, rcond=None)

    # Transpose b
    b_T = b.T

    b_T = b_T.reshape(height, width, -1)
    print(b_T.shape)

    return get_albedo_and_normal(b_T, height, width)


def uncalibrated_photometric_stereo(img_arr):
    width = img_arr.shape[2]
    height = img_arr.shape[1]
    i = img_arr.reshape(img_arr.shape[0], -1)
    i = np.transpose(i)
    print(i.shape)

    u, s, vh = np.linalg.svd(i, full_matrices=False)
    #print(u.shape)
    print(s)
    #print(vh.shape)
    s3 = np.eye(3)
    s3[0][0] = s[0]
    s3[1][1] = s[1]
    s3[2][2] = s[2]
    print(s3)

    b_star = np.matmul(u[:, :3], s3)
    b_star = b_star.reshape(height, width, 3)
    print(b_star.shape)

    return get_albedo_and_normal(b_star, height, width)


def get_albedo_and_normal(b, height, width):
    albedo_image = np.zeros(shape=(height, width))
    normal_map = np.zeros(shape=(height, width, 3))
    for i in range(height):
        for j in range(width):
            norm = np.linalg.norm(b[i][j], ord=2)

            # draw an albedo image
            albedo_image[i][j] = norm

            if norm != 0:
                normal_map[i][j] = b[i][j] / norm

                if not np.isnan(np.sum(normal_map[i][j])):
                    # draw an albedo image
                    albedo_image[i][j] = norm

    return albedo_image, normal_map
