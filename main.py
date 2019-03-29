from load_face_images import load_face_images
from photometric_stereo import photometric_stereo
import numpy as np
import matplotlib.pyplot as plt

# load data
amb_img, img_arr, light_dirs = load_face_images('./data/', 'yaleB01', 11)

# preprocess the data:
# subtract ambient_image from each image in imarray
file=open('temp.txt','w')
np.set_printoptions(threshold=np.inf)
img_arr = img_arr - amb_img
# make sure no pixel is less than zero
img_arr[img_arr < 0] = 0

# rescale values in imarray to be between 0 and 1
img_arr /= 255

#print(img_arr, file=file)
plt.imshow(img_arr[0])
plt.show()

albedo_img, normal_map = photometric_stereo(img_arr, light_dirs)

plt.imshow(albedo_img, cmap='gray')
plt.show()

print(normal_map.shape)
plt.imshow(normal_map)
plt.show()
