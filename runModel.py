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


K = 4
c, a, r_l = kmeans(pics_l, K, 100, sz)
print(c)
pp.pprint(a)
for i in range(K):
    plt.subplot(4, 3, i + 1)
    plt.imshow(c[i].reshape((H, W)), cmap='gray')
    plt.title("%.2f" % i)
plt.show()
