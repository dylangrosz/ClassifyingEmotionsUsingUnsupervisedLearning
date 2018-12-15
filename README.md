# Clustering Facial Emotion Using Unsupervised Learning 
GitHub: https://github.com/dylangrosz/ClassifyingEmotionsUsingUnsupervisedLearning
## Scripts
### runModel

This is the main script of our project. 
There are two main variables to be aware of when running this script, assuming you have the raw images in the correct folder. Line 167 sets the variables savedYet and toSave. Set toSave to True if you want to cache your feature extraction (advice: leave this set to True). Set savedYet to True if you do not want to extract and save features from your images and load from a previously cached copy; set it to False if you want to extract features from your set of images and cache it, assuming toSave is set to True. If starting this model from scratch, it is recommended to set savedYet to False and toSave to True. 
In terms of feature extraction, we have a set of boolean variables for each feature extraction method, which allows you to chose and combine any feature extraction method you want (e.g. literal pixels, HoG, DPM, HoG+DPM, etc.). The variables you can set in *featureExtract* work as follows:
- Setting *literal* to True flattens the pixels of the image and adds each pixel as a feature
- Setting *norm* to True subtracts each pixel by the mean pixel value and divides by the standard deviation before flattening. The resulting vector is added to the feature vector.
- Setting *hogF* to True adds the HoG feature vector to the overall feature vector.
- Setting *hogI* to True adds the flattened HoG image to the overall feature vector. 
- Setting *dpm* to True extracts the eyebrows, eyes, nose, and mouth regions and appends their flattened images into a single vector. This vector is then added to the overall feature vector.
- Setting *edge* to True applies a Canny edge detector to the image. The resulting image is then flattened and added to the feature vector.

Getting the extracted features, either by extracting during the runtime or loading from cache, will inevitably lead to kmeans clustering (further described in *ourKmeans*) and plotting of exemplars and success scoring statistics. We score our clusters by assigning each cluster the mode of its true labels (mimicking what a human would do to label a cluster based off of exemplars) and seeing how many points in the cluster are properly clustered.

### ourKmeans
We implemented 2 kmeans algorithms here: an optimized iterative kmeans (*kmeans*)and a vectorized kmeans (*kmeans_fast*). Based off of performance and memory capacity, our code runs on the iterative kmeans, or *kmeans*.

### labelImages
A simple script to assign the ground truth labels to the Cohn-Kanade dataset.

### edgeDetection
Our backend script to run edge detection on our images as a possible feature extractor. For the most part, this script acts as a Canny edge detector.

### detectFaceParts/images/detect_face_parts
A script that uses dlib and opencv2 to place a fixed dimension bounding rectangle around landmarks of various feacial features. Since we are dealing with emotion, we only care about the eyes, eyebrows, nose, and mouth.

### detectFaceParts\shape_predictor_68_face_landmarks.dat
This dat file acts as the static model for detect_face_parts to run on and landmark facial features. This is not included in the submission files because it is too large. Please refer to our GitHub to access this file.

## Dataset and Features
### data/cohn-kanade
This folder contains all of the raw image data from the Cohn-Kanade dataset. We only care about the final image for each session for each subject, for the nature of the dataset has each "session" of pictures start from a neutral face to a certain emotion. Line 41 of *runModel* details how to loop through each image in this folder. Of course, we only include the first few subjects to demonstrate the nature of the dataset.

### data/featureExtracted
This is where cached features go. Depending on how savedYet and toSave are set, *runModel* either saves from or loads to this folder.