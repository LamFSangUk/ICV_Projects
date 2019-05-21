import cv2
import numpy as np

import feature

def calculate_absolute_homography(H):
    for i in range(0,-1,-1):
        H[i] = np.dot(H[i + 1], H[i])
        H[i] = H[i] / H[i][2][2]

    for i in range(3,4,1):
        H[i] = np.dot(H[i-1], H[i ])
        H[i] = H[i] / H[i][2][2]

    return H

def leftshift(left_images):
    idx = 0
    res = None

    imgA = left_images[0]
    for imgB in left_images[1:]:
        H = feature.img_matching(imgA,imgB)

        H = np.linalg.inv(H)

        ds = np.dot(H, np.array([imgA.shape[1], imgA.shape[0], 1]))
        ds = ds / ds[-1]

        f1 = np.dot(H, np.array([0,0,1]))
        f1 = f1 / f1[-1]

        H[0][-1] += abs(f1[0])
        H[1][-1] += abs(f1[1])

        ds = np.dot(H, np.array([imgA.shape[1], imgA.shape[0], 1]))

        offset_y = abs(int(f1[1]))
        offset_x = abs(int(f1[0]))

        dsize = (int(ds[0])+offset_x, int(ds[1]) + offset_y)

        res = cv2.warpPerspective(imgA, H, dsize)

        #cv2.imshow("warped",res)
        #cv2.waitKey(0)

        res[offset_y:imgB.shape[0]+offset_y, offset_x: imgB.shape[1]+offset_x] = imgB

        imgA = res

        idx += 1

    return res

def rightshift(left_res, right_images):
    idx = 0
    res = None

    for img in right_images:
        H = feature.img_matching(left_res,img)

        txyz = np.dot(H, np.array([img.shape[1], img.shape[0], 1]))
        txyz = txyz/txyz[-1]

        dsize = (int(txyz[0])+left_res.shape[1], int(txyz[1])+left_res.shape[0])

        res = cv2.warpPerspective(img, H, dsize)

        res = mix_and_match(left_res, res)

        #cv2.imshow("warped", res)
        #cv2.waitKey(0)

        left_res = res

    return res

def mix_and_match(left_image, warped_image):
    i1y, i1x = left_image.shape[:2]
    #i2y, i2x = warped_image.shape[:2]

    #black_l = np.where(left_image == np.array([0,0,0]))
    #black_wi = np.where(warped_image == np.array([0,0,0]))

    for i in range(0, i1x):
        for j in range(0,i1y):
            try:
                if np.array_equal(left_image[j,i], np.array([0,0,0])) and np.array_equal(warped_image[j,i],np.array([0,0,0])):
                    warped_image[j,i] = [0,0,0]
                else:
                    if np.array_equal(warped_image[j,i],[0,0,0]):
                        warped_image[j,i] = left_image[j,i]
                    else:
                        if not np.array_equal(left_image[j][i], [0,0,0]):
                            #bw, gw, rw = warped_image[j,i]
                            bl, gl,rl = left_image[j,i]
                            warped_image[j,i] = [bl,gl,rl]

            except:
                pass


    return warped_image
