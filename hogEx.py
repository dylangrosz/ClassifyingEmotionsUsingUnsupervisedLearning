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


ex_fn = "data/face_03.png"
img = io.imread(ex_fn, as_grey=True)
hogFeature, hog_img = feature.hog(img, pixels_per_cell=(16, 16), visualise=True,
                                           feature_vector=True)
plt.subplot(1, 1, 1)
plt.imshow(hog_img, cmap='gray')
plt.show()