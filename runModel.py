'''
To install dlib:
    brew install cmake
    brew install boost
    pip install dlib
To install imutils:
    pip install --upgrade imutils
'''

import sys
from time import time
from collections import defaultdict
import numpy as np
import scipy
import matplotlib.pyplot as plt
from skimage import io, feature
import random
import _pickle as pkl
import pprint as pp
from tempfile import TemporaryFile
from ourKmeans import *
from face_recognition.examples.find_facial_features_in_picture import *

from os import listdir

data_fn = "data/cohn-kanade"
feature_fn = "data/featureExtracted"
img_ex_fn = "data/cohn-kanade/S010/001/S010_001_01594215.png"

pics = {}
pics_l = []

# To get to every image, use this loop:
# for subj in listdir(data_fn):
#     subj_fn = data_fn + "/" + subj
#     for sess in listdir(subj_fn):
#         sess_fn = subj_fn + "/" + sess
#         for p in listdir(sess_fn):
#             pic_fn = sess_fn + "/" + p
sz = io.imread(img_ex_fn, as_grey=True).flatten().shape[0]
H, W = io.imread(img_ex_fn, as_grey=True).shape

def dmp_featureExtract(image):
    # takes image and after adding a HOG feature of an
    # average left/right eye, mouth and nose, we pull out
    # the HOG and literal window of that area and add to
    # a flattened feature vector
    return []

def featureExtract(img, literal=True, norm=True, hog=True, dmp=True):
    features_p = np.array([])
    if literal:
        features_p = img.flatten()
    if norm:
        im_flat = img.flatten()
        features_p = (im_flat - np.mean(im_flat)) / np.std(im_flat)
    if hog:
        pixel_per_cell = 8
        hogFeature, hogImage = feature.hog(img, pixels_per_cell=(pixel_per_cell, pixel_per_cell), visualise=True,
                                           feature_vector=True)
        hog_flat = np.array(hogImage).flatten()
        #        features_p = np.append(features_p, hog_flat)
        features_p = np.append(features_p, hogFeature)
    if dmp:
        dmp_features = dmp_featureExtract(img)
    return features_p


# ADJUST THESE FOR SAVING AND REUSING
savedYet, toSave = True, True
cnt = 0
num_subj = 25
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
                        pic_f = featureExtract(img, literal=False)
                        if toSave:
                            with open(feature_fn + "/" + sess + "_" + sess_l[p_i] + "FE.pkl", 'wb') as handle:
                                #     pkl.dump(img, handle)
                                np.save(handle, img)
                        pics[subj][sess].append(img)
                        print(pic_f.shape)
                        pics_l.append(pic_f)

        print(cnt)
        cnt += 1
        if cnt >= num_subj:
            break
else:
    for f_n in listdir(feature_fn):
        with open(feature_fn + "/" + f_n, 'rb') as handle:
            # img = pkl.load(handle)
            img = np.load(handle).flatten()
            pics_l.append(img)

print("done")
pp.pprint(pics_l)


K = 4
c, a, r_l = kmeans(pics_l, K, 100, sz)
print(c)
pp.pprint(a)
center_assignments = {}
for i in a:
    k = a[i]
    if k in center_assignments:
        center_assignments[k].append(i)
    else:
        center_assignments[k] = [i]

pp.pprint(center_assignments)


for i in range(K):
    plt.subplot(4, 5, 3*i + 1)
#    plt.imshow(c[i][0:(H * W)].reshape((H, W)), cmap='gray')
    plt.imshow(pics_l[random.choice(center_assignments[i])].reshape((H, W)), cmap='gray')
    plt.title("%.2f" % i)
    plt.subplot(4, 3, i + 2)
    plt.imshow(pics_l[random.choice(center_assignments[i])].reshape((H, W)), cmap='gray')
    plt.title("%.2f" % i)
    plt.subplot(4, 3, i + 3)
    plt.imshow(pics_l[random.choice(center_assignments[i])].reshape((H, W)), cmap='gray')
    plt.title("%.2f" % i)
plt.show()
