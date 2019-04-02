from util import load_face_images, crop
from display_outputs import display_outputs
from photometric_stereo import *

output_dir = './outputs/'
num_image = 11

# load data
amb_img, img_arr, light_dirs = load_face_images('./data/', 'yaleB01', num_image)

# preprocess the data:
# subtract ambient_image from each image in imarray
img_arr = img_arr - amb_img
# make sure no pixel is less than zero
img_arr[img_arr < 0] = 0

# rescale values in imarray to be between 0 and 1
img_arr /= 255

# Crop the image so that only fae regions remain while the background and hair regions are excluded
img_arr = crop(img_arr, 255, 120, 180, 170)

# get albedo and normal
albedo_img, normal_map = photometric_stereo(img_arr, light_dirs)
(albedo_img_un, normal_map_un), light_dirs_un = uncalibrated_photometric_stereo(img_arr)

display_outputs(output_dir, num_image, img_arr, albedo_img, normal_map, light_dirs, albedo_img_un, normal_map_un, light_dirs_un)
