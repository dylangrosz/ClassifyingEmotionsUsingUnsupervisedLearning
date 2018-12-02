import sys
from time import time
from collections import defaultdict
import numpy as np
import scipy
import matplotlib.pyplot as plt
from skimage import io, feature, color
import random
import _pickle as pkl
import pprint as pp
from tempfile import TemporaryFile
from ourKmeans import *

# pip install opencv-python
import cv2
# pip install seam_carver
from seam_carver import intelligent_resize

from os import listdir

data_fn = "data/cohn-kanade"
feature_fn = "data/featureExtracted"
img_ex_fn = "data/cohn-kanade/S010/001/S010_001_01594215.png"

pics = {}
pics_l = []
pics_orig = []

# To get to every image, use this loop:
# for subj in listdir(data_fn):
#     subj_fn = data_fn + "/" + subj
#     for sess in listdir(subj_fn):
#         sess_fn = subj_fn + "/" + sess
#         for p in listdir(sess_fn):
#             pic_fn = sess_fn + "/" + p
sz = io.imread(img_ex_fn, as_gray=True).flatten().shape[0]
H, W = io.imread(img_ex_fn, as_gray=True).shape

def energy_function(image):
    H, W, _ = image.shape
    out = np.zeros((H, W))
    gray_image = color.rgb2gray(image)

    gradients = np.gradient(gray_image)
    yGradients = np.abs(gradients[0])
    xGradients = np.abs(gradients[1])
    out = yGradients + xGradients

    return out


def compute_cost(image, energy, axis=1):
    energy = energy.copy()

    if axis == 0:
        energy = np.transpose(energy, (1, 0))

    H, W = energy.shape

    cost = np.zeros((H, W))
    paths = np.zeros((H, W), dtype=np.int)

    cost[0] = energy[0]
    paths[0] = 0  # we don't care about the first row of paths

    for r in range(1,H):
        for c in range(W):
            up = cost[r-1][c]
            left = float('inf')
            right = float('inf')
            if c-1 >= 0:
                left = cost[r-1][c-1]
            if c+1 < W:
                right = cost[r-1][c+1]
            minEnergy = min(up, left, right)
            if minEnergy == up:
                cost[r][c] = up + energy[r][c]
                paths[r][c] = 0
            elif minEnergy == left:
                cost[r][c] = left + energy[r][c]
                paths[r][c] = -1
            elif minEnergy == right:
                cost[r][c] = right + energy[r][c]
                paths[r][c] = 1

    if axis == 0:
        cost = np.transpose(cost, (1, 0))
        paths = np.transpose(paths, (1, 0))

    # Check that paths only contains -1, 0 or 1
    assert np.all(np.any([paths == 1, paths == 0, paths == -1], axis=0)), \
           "paths contains other values than -1, 0 or 1"

    return cost, paths


def backtrack_seam(paths, end):
    H, W = paths.shape
    # initialize with -1 to make sure that everything gets modified
    seam = - np.ones(H, dtype=np.int)

    # Initialization
    seam[H-1] = end

    for r in range(1,H):
        direction = paths[H-r][seam[H-r]]
        seam[H-r-1] = seam[H-r] + direction

    # Check that seam only contains values in [0, W-1]
    assert np.all(np.all([seam >= 0, seam < W], axis=0)), "seam contains values out of bounds"

    return seam


def remove_seam(image, seam):
    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=2)

    out = None
    H, W, C = image.shape
    if C > 1:
        out = np.zeros((H,W-1,C))
    else:
        out = np.zeros((H,W-1))
    skipped = False
    for r in range(H):
        for c in range(W):
            if seam[r] != c:
                if skipped:
                    out[r][c-1] = image[r][c]
                else:
                    out[r][c] = image[r][c]
            else:
                skipped = True
        skipped = False
    out = out.astype(image.dtype)
    out = np.squeeze(out)  # remove last dimension if C == 1

    # Make sure that `out` has same type as `image`
    assert out.dtype == image.dtype, \
       "Type changed between image (%s) and out (%s) in remove_seam" % (image.dtype, out.dtype)

    return out


def reduce(image, size, axis=1, efunc=energy_function, cfunc=compute_cost):
    out = np.copy(image)
    if axis == 0:
        out = np.transpose(out, (1, 0, 2))

    H = out.shape[0]
    W = out.shape[1]

    assert W > size, "Size must be smaller than %d" % W

    assert size > 0, "Size must be greater than zero"

    pixelsToRemove = W - size
    for i in range(pixelsToRemove):
        print(i)
        energy = efunc(out)
        cost, paths = cfunc(out, energy)
        end = np.argmin(cost[-1])
        seam = backtrack_seam(paths, end)
        out = remove_seam(out, seam)

    assert out.shape[1] == size, "Output doesn't have the right shape"

    if axis == 0:
        out = np.transpose(out, (1, 0, 2))

    return out

def norm_and_flatten(img):
    im_flat = img.flatten()
    return (im_flat - np.mean(im_flat)) / np.std(im_flat)

#ADJUST THESE FOR SAVING AND REUSING
savedYet, toSave = True, True
cnt = 0
num_subj = 50
if not savedYet:
    for subj in listdir(data_fn):
        subj_fn = data_fn + "/" + subj
        pics[subj] = {}
        for sess in listdir(subj_fn):
            sess_fn = subj_fn + "/" + sess
            pics[subj][sess] = []
            sess_l = listdir(sess_fn)
            for p_i in range(len(sess_l)):
                if p_i == len(listdir(sess_fn)) - 1:
                    pic_fn = sess_fn + "/" + sess_l[p_i]
                    img = io.imread(pic_fn, as_gray=True)
                    H_i, W_i = img.shape
                    if H_i == H and W_i == W:
                        img = norm_and_flatten(img)
                        if toSave:
                            with open(feature_fn + "/" + sess + "_" + sess_l[p_i] + "FE.pkl", 'wb') as handle:
                            #     pkl.dump(img, handle)
                                np.save(handle, img)
                        pics[subj][sess].append(img)
                        pics_l.append(img)

        print(cnt)
        cnt += 1
        if cnt >= num_subj:
            break
else:
    for f_n in listdir(feature_fn):
        with open(feature_fn + "/" + f_n, 'rb') as handle:
            #print("go")
            #img = pkl.load(handle)
            img = np.load(handle)
            pics_l.append(img)

print("done")
pp.pprint(pics)

'''
cnt = 0
num_subj = 10
for subj in listdir(data_fn):
    subj_fn = data_fn + "/" + subj
    pics[subj] = {}
    for sess in listdir(subj_fn):
        sess_fn = subj_fn + "/" + sess
        pics[subj][sess] = []
        sess_l = listdir(sess_fn)
        for p_i in range(len(sess_l)):
            if p_i == len(listdir(sess_fn)) - 1:
                pic_fn = sess_fn + "/" + sess_l[p_i]

                img = cv2.imread(pic_fn,0)
                edges = cv2.Canny(img,60,150)
                print(pic_fn)

                H, W = edges.shape
                edges = color.gray2rgb(edges)
                #print(W//2)
                #edges = reduce(edges,W//2)

                rgb_weights = [0, 0, 0]
                mask_weight = 10
                mask = np.zeros(edges.shape)

                edges = intelligent_resize(edges, 0, -W//2, rgb_weights, mask, mask_weight)

                edges = color.rgb2gray(edges)

                #plt.subplot(121),plt.imshow(img,cmap = 'gray')
                #plt.title('Original Image'), plt.xticks([]), plt.yticks([])
                #plt.subplot(122),plt.imshow(edges,cmap = 'gray')
                #plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

                #plt.show()

                #img = io.imread(pic_fn, as_gray=True)
                H_i, W_i = edges.shape
                if H_i == H and W_i == W:
                    edges = norm_and_flatten(edges)
                    pics_orig.append(img)
                    pics[subj][sess].append(edges)
                    pics_l.append(edges)
    #print(cnt)
    cnt += 1
    if cnt >= num_subj: break
#pp.pprint(pics)
'''


K = 4
c, a, r_l = kmeans(pics_l, K, 100, sz)
print(c)
pp.pprint(a)
for i in range(K):
    plt.subplot(4, 3, i + 1)
    plt.imshow(c[i].reshape((H, W)), cmap='gray')
    plt.title("%.2f" % i)
plt.show()
'''
K = 5
c, a, r_l = kmeans_fast(pics_l, K, 100)
#print(c)
pp.pprint(a)
for i in range(len(a)):
    if a[i] == 0:
        plt.subplot(121),plt.imshow(pics_orig[i],cmap = 'gray')
        plt.title('0'), plt.xticks([]), plt.yticks([])
        plt.show()
    if a[i] == 1:
        plt.subplot(121),plt.imshow(pics_orig[i],cmap = 'gray')
        plt.title('1'), plt.xticks([]), plt.yticks([])
        plt.show()
    if a[i] == 2:
        plt.subplot(121),plt.imshow(pics_orig[i],cmap = 'gray')
        plt.title('2'), plt.xticks([]), plt.yticks([])
        plt.show()
    if a[i] == 3:
        plt.subplot(121),plt.imshow(pics_orig[i],cmap = 'gray')
        plt.title('3'), plt.xticks([]), plt.yticks([])
        plt.show()
    if a[i] == 4:
        plt.subplot(121),plt.imshow(pics_orig[i],cmap = 'gray')
        plt.title('4'), plt.xticks([]), plt.yticks([])
        plt.show()

#for i in range(K):
#    plt.subplot(4, 3, i + 1)
#    plt.imshow(c[i].reshape((H, W)), cmap='gray')
#    plt.title("%.2f" % i)
plt.show()
'''
