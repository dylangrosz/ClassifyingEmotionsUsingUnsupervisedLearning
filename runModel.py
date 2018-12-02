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


def kmeans_fast(features, k, num_iters=100):
    """ Use kmeans algorithm to group features into k clusters.

    This function makes use of numpy functions and broadcasting to speed up the
    first part(cluster assignment) of kmeans algorithm.

    Hints
    - You may find np.repeat and np.argmin useful

    Args:
        features - Array of N features vectors. Each row represents a feature
            vector.
        k - Number of clusters to form.
        num_iters - Maximum number of iterations the algorithm will run.

    Returns:
        assignments - Array representing cluster assignment of each point.
            (e.g. i-th point is assigned to cluster assignments[i])
    """
    N = len(features)
    # Randomly initalize cluster centers
    centers, assignments = np.array([features[i] for i in sorted(random.sample(range(0, len(features)), K))]), {}
    _, dim = centers.shape

    tile_f = np.tile(features, (k, 1))
    for n in range(num_iters):
        tile_c = np.repeat(centers, N, axis=0)
        tile_sub = np.subtract(tile_f, tile_c)
        dist = np.linalg.norm(tile_sub, axis=1).reshape(k, N)
        assignments_new = np.argmin(dist, axis=0)
        if np.array_equal(assignments, assignments_new):
            break
        else:
            assignments = assignments_new

        old_centers, new_stats = centers[:], []
        for j in range(0, k):
            new_stats.append([0, []])
        for i in range(len(features)):
            c_ind = int(assignments[i])
            num, sum_vec = new_stats[c_ind]
            sum_vec_new = np.add(sum_vec, features[i]) if len(sum_vec) > 0 else features[i]
            new_stats[c_ind] = [num + 1, sum_vec_new]

        new_centers = np.zeros((k, dim))
        for j in range(0, k):
            num, new_center = new_stats[j]
            new_center = np.divide(new_center, num)
            new_centers[j] = np.array(new_center)
        centers = new_centers

    return centers, assignments, 0


K = 6
c, a, r_l = kmeans(pics_l, K, 100)
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
