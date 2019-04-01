from util import load_face_images, crop, normal2rgb, validate_with_specific_light
from photometric_stereo import *
import matplotlib.pyplot as plt

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

# save the images from photometric stereo
plt.imshow(albedo_img, cmap='gray')
plt.title('albedo')
plt.savefig(output_dir + 'albedo.png')

normal_rgb = normal2rgb(normal_map)
plt.imshow(normal_rgb)
plt.title('normal map')
plt.savefig(output_dir + 'normal_map.png')

# save the images from uncalibrated photometric stereo
plt.imshow(albedo_img_un, cmap='gray')
plt.title('albedo uncalibrated')
plt.savefig(output_dir + 'albedo_uncalibrated.png')

normal_rgb_un = normal2rgb(normal_map_un)
plt.imshow(normal_rgb_un)
plt.title('normal map uncalibrated')
plt.savefig(output_dir + 'normal map_uncalibrated.png')
plt.clf()

f, axarr = plt.subplots(2, 2)
axarr[0, 0].imshow(albedo_img, cmap='gray')
axarr[0, 1].imshow(normal_rgb)
axarr[1, 0].imshow(albedo_img_un, cmap='gray')
axarr[1, 1].imshow(normal_rgb_un)
plt.show()
plt.clf()

# validate photometric_stereo
f, axarr = plt.subplots(3, num_image)
for i in range(num_image):
    re_img = validate_with_specific_light(normal_map, albedo_img, light_dirs[i])
    re_img_un = validate_with_specific_light(normal_map_un, albedo_img_un, light_dirs_un[i])
    axarr[0, i].imshow(img_arr[i], cmap='gray')
    axarr[1, i].imshow(re_img, cmap='gray')
    axarr[2, i].imshow(re_img_un, cmap='gray')

plt.show()
