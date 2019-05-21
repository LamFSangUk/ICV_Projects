import numpy as np
import cv2

import feature
import stitch

# load images
img = []
img.append(cv2.resize(cv2.imread('input/indoor_302_01.jpg',cv2.IMREAD_COLOR),dsize=(256,256)))
img.append(cv2.resize(cv2.imread('input/indoor_302_02.jpg',cv2.IMREAD_COLOR),dsize=(256,256)))
img.append(cv2.resize(cv2.imread('input/indoor_302_03.jpg',cv2.IMREAD_COLOR),dsize=(256,256)))
img.append(cv2.resize(cv2.imread('input/indoor_302_04.jpg',cv2.IMREAD_COLOR),dsize=(256,256)))
img.append(cv2.resize(cv2.imread('input/indoor_302_05.jpg',cv2.IMREAD_COLOR),dsize=(256,256)))

for i in range(4):
    feature.img_matching(img[i],img[i+1],show=True)

# Stiching
res = stitch.leftshift(img[0:3])
print("finish left")
#cv2.imshow("half", res)
#cv2.waitKey(0)
res = stitch.rightshift(res, img[3:5])

cv2.imwrite("output.jpg", res)
cv2.waitKey(0)