import numpy as np

def photometric_stereo(img_arr, light_dirs):
    width = img_arr.shape[2]
    height = img_arr.shape[1]
    print(np.shape(img_arr))
    print(np.shape(light_dirs))
    # light_dir shape : num_imagesX3
    # image_arr : num_imageX(heightXwidth)
    img_arr = img_arr.reshape(img_arr.shape[0], -1)
    print(img_arr.shape)

    light_dirs_T = light_dirs.transpose()
    b = np.matmul(np.matmul(np.linalg.inv(np.matmul(light_dirs_T, light_dirs)), light_dirs_T), img_arr)

    # Transpose b
    b_T = b.transpose()

    b_T = b_T.reshape(height, width, -1)
    print(b_T.shape)

    albedo_image = np.zeros(shape=(height, width))
    normal_map = np.zeros(shape=(height, width, 3))
    for i in range(height):
        for j in range(width):
            norm = np.linalg.norm(b_T[i][j], ord=2)

            # draw an albedo image
            albedo_image[i][j] = norm

            if norm != 0:
                normal_map[i][j] = b_T[i][j] / norm
                #print(normal_map[i][j])

    return albedo_image, normal_map
