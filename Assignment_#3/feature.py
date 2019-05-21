import numpy as np
import cv2

def img_matching(imgA, imgB, show=False):
    img_gray1 = cv2.cvtColor(imgA, cv2.COLOR_BGR2GRAY)
    img_gray2 = cv2.cvtColor(imgB, cv2.COLOR_BGR2GRAY)

    org_kps1, kps1, des1 = feature_extraction(img_gray1)
    org_kps2, kps2, des2 = feature_extraction(img_gray2)

    matches, H = feature_matching(kps1, kps2, des1, des2, 0.7)

    if show is True:
        img = cv2.drawMatchesKnn(imgA,org_kps1,imgB,org_kps2,matches,None,flags=2)
        cv2.imshow("matching",img)
        cv2.waitKey(0)

    return H

def feature_extraction(img_gray):
    sift = cv2.xfeatures2d.SIFT_create(200)

    kps, descriptors = sift.detectAndCompute(img_gray, None)

    np_kps = np.array([kp.pt for kp in kps])

    return kps, np_kps, descriptors

def feature_matching(kps1, kps2, des1, des2, ratio):

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params,search_params)
    raw_matches = flann.knnMatch(des1, des2, k=2)

    matches = []
    for m in raw_matches:
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            matches.append((m[0].trainIdx, m[0].queryIdx))

    if len(matches) > 4:

        ptsA = np.float32([kps1[i] for (_, i) in matches])
        ptsB = np.float32([kps2[i] for (i, _) in matches])

        #(H, status) = cv2.findHomography(ptsB, ptsA, cv2.RANSAC,4.0)
        H_, error = h_by_ransac(ptsB,ptsA)

        #print(H)
        print("H=",H_, error)


    return raw_matches, H_

def h_by_ransac(ptsA, ptsB):

    idx_arr = np.arange(len(ptsA))
    H = np.empty((3, 3))
    total_sum_error = 0
    max_num_inliers = 0

    for i in range(1000):

        idx = np.random.choice(idx_arr, 4, replace=False)

        sampleA = ptsA[idx]
        sampleB = ptsB[idx]

        # DLT
        H_candidate = dlt(sampleA, sampleB)

        dist = calculate_transfer_error(ptsA,ptsB,H_candidate)
        total_error = np.sum(dist)
        num_inliers = len(dist[dist<1.25])

        if max_num_inliers < num_inliers:
            H = H_candidate
            max_num_inliers = num_inliers
            total_sum_error = total_error

    return H, total_sum_error

def dlt(p1, p2):
    A = []
    for i in range(0, len(p1)):
        x, y = p1[i][0], p1[i][1]
        u, v = p2[i][0], p2[i][1]
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y, -u])
        A.append([0, 0, 0, x, y, 1, -v * x, -v * y, -v])
    A = np.asarray(A)
    U, S, Vh = np.linalg.svd(A)
    L = Vh[-1, :] / Vh[-1, -1]
    H = L.reshape(3, 3)

    return H

def calculate_transfer_error(p1, p2, H):
    # padding
    p1_pad = np.pad(p1, ((0,0),(0,1)), mode='constant', constant_values=1)
    p1_pad = np.transpose(p1_pad)
    p2_pad = np.pad(p2, ((0,0),(0,1)), mode='constant', constant_values=1)
    p2_pad = np.transpose(p2_pad)

    H_inv = np.linalg.inv(H)

    p1_move = np.dot(H, p1_pad)
    p1_move = np.transpose(p1_move)
    p1_move = np.true_divide(p1_move[:, :2], p1_move[:, [-1]])
    p2_move = np.dot(H_inv, p2_pad)
    p2_move = np.transpose(p2_move)
    p2_move = np.true_divide(p2_move[:, :2], p2_move[:, [-1]])

    dist1 = p1 - p2_move
    dist2 = p2 - p1_move

    dist1 = dist1 ** 2
    dist2 = dist2 ** 2

    dist = dist1+dist2

    return dist
