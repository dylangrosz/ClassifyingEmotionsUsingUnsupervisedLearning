'''
To install dlib:
    brew install cmake
    brew install boost
    pip install dlib
To install imutils:
    pip install --upgrade imutils

pip install opencv-python
'''

import sys
from time import time
from collections import defaultdict
import numpy as np
import scipy
import matplotlib.pyplot as plt
from skimage import io, feature
import random
import pprint as pp
from ourKmeans import *
from detectFaceParts import *
# import necessary packages for detecting face parts
from imutils import face_utils
import argparse
import imutils
import dlib
import cv2
import csv

from os import listdir

data_fn = "data/cohn-kanade"
feature_fn = "data/hogFeatureI"
#feature_fn_folders = ["data/hogFeatureF"]
feature_fn_folders = ["data/hogFeatureI"]#, "data/normFeature"]
img_ex_fn = "data/cohn-kanade/S010/001/S010_001_01594215.png"
labels = "data/labels.csv"

pics = {}
pics_f, pics_literal = [], []

# To get to every image, use this loop:
# for subj in listdir(data_fn):
#     subj_fn = data_fn + "/" + subj
#     for sess in listdir(subj_fn):
#         sess_fn = subj_fn + "/" + sess
#         for p in listdir(sess_fn):
#             pic_fn = sess_fn + "/" + p
sz = io.imread(img_ex_fn, as_grey=True).flatten().shape[0]
H, W = io.imread(img_ex_fn, as_grey=True).shape
emotions_dict = {}
emotions_labels = []

f = open(labels, encoding='utf-8-sig')
reader = csv.reader(f)
for row in reader:
    print(row)
    if (len(row[0]) != 3):
        if row[0][-2:] == "10" and row[1] == "001":
            row[0] = "010"
        else:
            row[0] = "0" + row[0]
    #print(row[0] + "_" + row[1])
    emotions_dict[row[0] + "_" + row[1]] = row[2]
emotions_dict["094_002"] = "na"
emotions_dict["100_001"] = "na"

def dpm_featureExtract(image):
    features = np.array([])
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("detectFaceParts/shape_predictor_68_face_landmarks.dat")

    gray = image
    #edges = cv2.Canny(image ,60,200)

    #gray = imutils.resize(image, width=500)
    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 1)

    for (i, rect) in enumerate(rects):
    # determine the facial landmarks for the face region, then
    # convert the landmark (x, y)-coordinates to a NumPy array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        #edges = cv2.Canny(image,50,160)

        # loop over the face parts individually
        forehead_topleft = (0,0)
        forehead_topright = (0,0)
        forehead_bottomleft = (0,0)
        forehead_bottomright = (0,0)
        for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
            # clone the original image so we can draw on it, then
            # display the name of the face part on the image
            clone = image.copy()#edges.copy()
            cv2.putText(clone, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 0, 255), 2)

            # loop over the subset of facial landmarks, drawing the
            # specific face part
            for (x, y) in shape[i:j]:
                cv2.circle(clone, (x, y), 1, (0, 0, 255), -1)

            # extract the ROI of the face region as a separate image
            (x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
            roi = image[y:y + h, x:x + w]
            print(name)
            if name == "right_eyebrow":
                forehead_topright = (y,x)
            if name == "left_eyebrow":
                forehead_topleft = (y,x+w)
            if name == "right_eye":
                forehead_bottomright = (y+h,x)
            if name == "left_eye":
                forehead_bottomleft = (y+h,x+w)
                #print(min(forehead_topleft[0],forehead_topright[0]))
                #print(max(forehead_bottomleft[0],forehead_bottomright[0]))
                #print(min(forehead_bottomleft[1],forehead_topleft[1]))
                #print(max(forehead_bottomright[1],forehead_topright[1]))
                roi = image[min(forehead_topleft[0],forehead_topright[0]):max(forehead_bottomleft[0],forehead_bottomright[0]),
                        max(forehead_bottomright[1],forehead_topright[1]):min(forehead_bottomleft[1],forehead_topleft[1])]
            #roi = imutils.resize(roi, height=500, width=250, inter=cv2.INTER_CUBIC)
            if name == "nose":
                roi = cv2.resize(roi, (40,80))
                roi = (roi - np.mean(roi)) / np.std(roi)
            if name == "mouth" or name == "left_eye":
                roi = cv2.resize(roi, (100,33))
            # add facial feature
            #print(roi.flatten())
                roi = (roi - np.mean(roi)) / np.std(roi)

            if name == "mouth" or name == "left_eye" or name == "nose":
                #cv2.imshow("ROI", roi)
                #cv2.waitKey(0)
                features = np.append(features,roi.flatten())
            #print(features)

    #print(features)
    return features

def featureExtract(img, literal=True, norm=True, hogF=True, hogI=True, dpm=True, edge=True):
    features_p = np.array([])
    if literal:
        features_p = img.flatten()
    if norm:
        im_flat = img.flatten()
        features_p = (im_flat - np.mean(im_flat)) / np.std(im_flat)
    if hogF or hogI:
        pixel_per_cell = 8
        hogFeature, hogImage = feature.hog(img, pixels_per_cell=(pixel_per_cell, pixel_per_cell), visualise=True,
                                           feature_vector=True)
        hog_flat = np.array(hogImage).flatten()
        if hogI:
            features_p = np.append(features_p, hog_flat)
        if hogF:
            features_p = np.append(features_p, hogFeature)
    if dpm:
        dpm_features = dpm_featureExtract(img)
        #print(dpm_features)
        features_p = np.append(features_p, dpm_features)
    if edge:
        edges_features = cv2.Canny(img,60,150)
        cv2.imshow("Image", edges_features)
        cv2.waitKey(0)
        features_p = np.append(features_p, edges_features.flatten())
    return features_p

# ADJUST THESE FOR SAVING AND REUSING
savedYet, toSave = False, False
cnt = 0
num_subj = 1000
if not savedYet:
    for subj in listdir(data_fn):
        subj_fn = data_fn + "/" + subj
        pics[subj] = {}
        for sess in listdir(subj_fn):
            sess_fn = subj_fn + "/" + sess
            emotions_labels.append(emotions_dict[subj_fn[-3:] + "_" + sess])
            pics[subj][sess] = []
            sess_l = listdir(sess_fn)
            for p_i in range(len(sess_l)):
                if p_i == len(listdir(sess_fn)) - 1:
                    pic_fn = sess_fn + "/" + sess_l[p_i]
                    img = io.imread(pic_fn, as_grey=True)
                    H_i, W_i = img.shape
                    if H_i == H and W_i == W:
                        pic_f = featureExtract(img, literal=False, norm=False, hogF=False, hogI=False, dpm=False, edge=True)
                        if toSave:
                            with open(feature_fn + "/" + sess + "_" + sess_l[p_i][:-4] + "_FE", 'wb') as handle:
                                np.save(handle, pic_f)
                        pics[subj][sess].append(img)
                        print(pic_f.shape)
                        pics_f.append(pic_f)
                        pics_literal.append(img)
                        sz = pic_f.shape[0]

        print(cnt)
        cnt += 1
        if cnt >= num_subj:
            break
else:
    for f_n in listdir(feature_fn):
        '''
        with open(feature_fn + "/" + f_n, 'rb') as handle:
            p_feature = np.load(handle).flatten()
            pics_f.append(p_feature)
            sz = p_feature.shape[0]
            emotions_labels.append(emotions_dict[f_n[5:12]])
        '''
        #p_feature = np.array([])
        p_feature = np.array([])
        for folder in feature_fn_folders:
            handle = open(folder + "/" + f_n, 'rb')
                # img = pkl.load(handle)
            curr_feature = np.array(np.load(handle).flatten())
            p_feature = np.append(p_feature, curr_feature)
            #np.append(p_feature, np.load(handle).flatten())

        pics_f.append(p_feature)
        #print(p_feature)
        sz = p_feature.shape[0]
        emotions_labels.append(emotions_dict[f_n[5:12]])
    for subj in listdir(data_fn):
        subj_fn = data_fn + "/" + subj
        pics[subj] = {}
        for sess in listdir(subj_fn):
            sess_fn = subj_fn + "/" + sess
            emotions_labels.append(emotions_dict[subj_fn[-3:] + "_" + sess])
            pics[subj][sess] = []
            sess_l = listdir(sess_fn)
            for p_i in range(len(sess_l)):
                if p_i == len(listdir(sess_fn)) - 1:
                    pic_fn = sess_fn + "/" + sess_l[p_i]
                    img = io.imread(pic_fn, as_gray=True)
                    pics_literal.append(img)

print("done")
pp.pprint(pics_f)

K = 6
c, a, r_l = kmeans(pics_f, K, 100, sz)
#print(c)
#print(a)
center_assignments = {}
center_emotions = {}
for i in a:
    k = a[i]
    if k in center_assignments:
        center_assignments[k].append(i)
        center_emotions[k].append(emotions_labels[i])
    else:
        center_assignments[k] = [i]
        center_emotions[k] = [emotions_labels[i]]

#print(center_assignments)
#print(center_emotions)

cluster_labels = {}
for k in center_emotions:
    cluster_labels[k] = max(set(center_emotions[k]), key=center_emotions[k].count)

#print(cluster_labels)
total_mislabel, total_ex = 0, 0
for c in cluster_labels:
    emotion = cluster_labels[c]
    mislabel = 0
    for label in center_emotions[c]:
        if label != emotion and label != "na":
            mislabel += 1
            total_mislabel += 1
        total_ex = total_ex + 1 if label != "na" else total_ex
    print("Success rate for " + emotion + ": " + str(1 - (mislabel / len(center_emotions[c]))))
print("Overall success rate: " + str(1 - (total_mislabel / total_ex)))
plt_assignments = center_assignments.copy()
for i in range(K):
    if i in center_assignments:
        num_exemplars = 9
        for iter_img in range(num_exemplars):
            if iter_img < len(plt_assignments[i]):
                plt.subplot(3, 3, iter_img + 1)
                r_choice = random.choice(plt_assignments[i])
                #print(pics_literal[r_choice].shape)
                plt.imshow(pics_literal[r_choice].reshape((H, W)), cmap='gray')
                ind = plt_assignments[i].index(r_choice)
                plt_assignments[i].pop(ind)
                plt.title("%.2f" % i)
        plt.show()
