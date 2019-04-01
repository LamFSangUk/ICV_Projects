from util import load_face_images, crop, normal2rgb
from photometric_stereo import *
import numpy as np
import matplotlib.pyplot as plt

# load data
amb_img, img_arr, light_dirs = load_face_images('./data/', 'yaleB01', 11)

# preprocess the data:
# subtract ambient_image from each image in imarray
img_arr = img_arr - amb_img
# make sure no pixel is less than zero
img_arr[img_arr < 0] = 0

# rescale values in imarray to be between 0 and 1
img_arr /= 255


# Crop the image so that only fae regions remain while the background and hair regions are excluded
img_arr = crop(img_arr, 255, 120, 180, 170)
plt.imshow(img_arr[-2], cmap='gray')
plt.show()

# get albedo and normal
#albedo_img, normal_map = photometric_stereo(img_arr, light_dirs)
albedo_img, normal_map = uncalibrated_photometric_stereo(img_arr)

plt.imshow(albedo_img, cmap='gray')
plt.show()

normal_rgb = normal2rgb(normal_map)
plt.imshow(normal_rgb)
plt.show()


def validate_with_specific_light(normal_map, albedo_img, light_dir):
    img = np.dot(normal_map, light_dir)
    img = np.multiply(albedo_img, img)

    img *= 255
    # img +=
    img[img < 0] = 0
    print(img.shape)

    print(light_dir)

    return img


re_img = validate_with_specific_light(normal_map, albedo_img, [0, 0, -1])
plt.imshow(re_img, cmap='gray')
plt.show()
