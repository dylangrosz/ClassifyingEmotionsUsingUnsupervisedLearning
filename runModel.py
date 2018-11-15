import numpy as np
import sys
from time import time
from collections import defaultdict

import numpy as np
import scipy
import matplotlib.pyplot as plt
from matplotlib import rc
from skimage import io
import random
import pprint as pp

from os import listdir

data_fn = "data/cohn-kanade"
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
cnt = 0
num_subj = 20
for subj in listdir(data_fn):
    subj_fn = data_fn + "/" + subj
    pics[subj] = {}
    for sess in listdir(subj_fn):
        sess_fn = subj_fn + "/" + sess
        pics[subj][sess] = []
        for p in listdir(sess_fn):
            pic_fn = sess_fn + "/" + p
            img = io.imread(pic_fn, as_gray=True)
            H_i, W_i = img.shape
            if H_i == H and W_i == W:
                pics[subj][sess].append(img.flatten())
                pics_l.append(img.flatten())
    cnt += 1
    if cnt >= num_subj: break
pp.pprint(pics)


def kmeans(examples, K, maxIters):
    '''
    examples: list of examples, each example is a string-to-double dict representing a sparse vector.
    K: number of desired clusters. Assume that 0 < K <= |examples|.
    maxIters: maximum number of iterations to run (you should terminate early if the algorithm converges).
    Return: (length K list of cluster centroids,
            list of assignments (i.e. if examples[i] belongs to centers[j], then assignments[i] = j)
            final reconstruction loss)
    '''
    # BEGIN_YOUR_CODE (our solution is 32 lines of code, but don't worry if you deviate from this)
    centers, assignments = [examples[i] for i in sorted(random.sample(range(0, len(examples)), K))], {}
    prev_rec_loss, rec_loss = -1, 0

    # ||u-x_i||^2=uTu-2uTx_i +x_iTx_i
    # Precalculate x_iTx_i portion, only need to do this once
    xTx = []
    for i in range(0, len(examples)):
        xTx.append(np.dot(examples[i], examples[i]))

    while maxIters > 0:
        # Precalculate uTu portion, only need to do this once per iteration
        uTu = []
        for j in range(0, K):
            uTu.append(np.dot(centers[j], centers[j]))

        # Assign
        rec_loss = 0
        new_centers = []
        for j in range(0, K):
            new_centers.append([0, np.zeros(sz)])
        for i in range(0, len(examples)):
            rec_losses = []
            for j in range(0, K):
                uTx_i = np.dot(centers[j], examples[i])
                rec_losses.append((uTu[j] - 2 * uTx_i + xTx[i], j))
            min_loss, ind = min(rec_losses)
            rec_loss += min_loss
            assignments[i] = ind
            N, sum_vec = new_centers[ind]
            sum_vec_new = sum_vec + examples[i]
            new_centers[ind] = (N + 1, sum_vec_new)

        # Recalculate
        for j in range(0, K):
            N, new_center = new_centers[j]
            scale = 1 / float(N) if N > 0 else 0
            new_center = np.multiply(new_center, scale)
            centers[j] = new_center
        if rec_loss == prev_rec_loss:
            print("Converged at " + str(rec_loss))
            break
        print("Reconstruction Loss at Iter #" + str(maxIters) + " = " + str(rec_loss))
        prev_rec_loss = rec_loss
        maxIters -= 1
    return centers, assignments, rec_loss

K = 2
c, a, r_l = kmeans(pics_l, K, 100)
print(c)
pp.pprint(a)
for i in range(K):
    plt.subplot(4, 3, i + 1)
    plt.imshow(c[i].reshape((H, W)), cmap='gray')
    plt.title("%.2f" % i)
    # ks = [k for k,v in a if v == i]
    # plt.imshow(pics_l[random.choice(ks)].reshape((H, W)))
plt.show()
