import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.io
import time
import cv2
import numpy as np

from ex1_functions import *


def tic():
    return time.time()


def toc(t):
    return float(tic()) - float(t)


##########################################################
# Don't forget to fill in your IDs!!!
# students' IDs:
ID1 = 312739485
ID2 = 300016417
##########################################################


# Parameters
max_err = 25
inliers_percent = 0.8

show_H=True

# Read the data:
img_src = mpimg.imread('src.jpg')
img_dst = mpimg.imread('dst.jpg')
matching = scipy.io.loadmat('matches')  # matching points and some outliers
perfect_matching = scipy.io.loadmat('matches_perfect')  # loading perfect matches


#####################
#### The problem ####
#####################

# Display the matching points on both images and check if they are indeed a perfect match
display_matching(src_image=img_src, dest_image=img_dst, perfect_matching=create_matching_dict(perfect_matching),
                 matching=create_matching_dict(matching))


########################################
#### Part A: Homography computation ####
########################################

# Compute naive homography
match_p_src, match_p_dst=perfect_matching['match_p_src'],perfect_matching['match_p_dst']
tt = time.time()
H_naive = compute_homography_naive(match_p_src, match_p_dst)
print('Naive Homography - matching perfect {:5.4f} sec'.format(toc(tt)))
print(H_naive)


#Forward Mapping
width = img_src.shape[1] + img_dst.shape[1]
height = img_src.shape[0] + img_dst.shape[0]
if show_H:
    src_forward_mapping = cv2.warpPerspective(img_src, H_naive, dsize=(width,height))
    plt.figure()
    panplot = plt.imshow(src_forward_mapping)
    plt.title('Source Image - matches_perfect')
    plt.show()


#Repeat for matching
match_p_src, match_p_dst=matching['match_p_src'],matching['match_p_dst']
tt = time.time()
H_naive = compute_homography_naive(match_p_src, match_p_dst)
print('Naive Homography - matching {:5.4f} sec'.format(toc(tt)))
print(H_naive)

if show_H:
    src_forward_mapping = cv2.warpPerspective(img_src, H_naive, dsize=(width,height))
    plt.figure()
    panplot = plt.imshow(src_forward_mapping)
    plt.title('Source Image - matching')
    plt.show()


#########################################
##### Part B: Dealing with outliers #####
#########################################


# Test naive homography
tt = time.time()
fit_percent, dist_mse = test_homography(H_naive, match_p_src, match_p_dst, max_err)
print('Naive Homography Test {:5.4f} sec'.format(toc(tt)))
print([fit_percent, dist_mse])

# Compute RANSAC homography
tt = tic()
H_ransac = compute_homography(match_p_src, match_p_dst, inliers_percent, max_err)
print('RANSAC Homography {:5.4f} sec'.format(toc(tt)))
print(H_ransac)

# Test RANSAC homography
tt = tic()
fit_percent, dist_mse = test_homography(H_ransac, match_p_src, match_p_dst, max_err)
print('RANSAC Homography Test {:5.4f} sec'.format(toc(tt)))
print([fit_percent, dist_mse])

if show_H:
    src_forward_mapping = cv2.warpPerspective(img_src, H_ransac, dsize=(width, height) )
    plt.figure()
    panplot = plt.imshow(src_forward_mapping)
    plt.title('Source Image - matches_perfect')
    plt.show()


#########################################
####### Part C: Panorama creation #######
#########################################


#backward mapping with linear interpolation
if show_H:
    src_backward_mapping = cv2.warpPerspective(img_src, H_ransac, dsize=(width, height),flags=cv2.WARP_INVERSE_MAP|cv2.INTER_LINEAR)
    plt.figure()
    panplot = plt.imshow(src_backward_mapping)
    plt.title('backward mapping with linear interpolation')
    plt.show()


# Build panorama
tt = tic()
img_pan = panorama(img_src, img_dst, match_p_src, match_p_dst, inliers_percent, max_err)
print('Panorama {:5.4f} sec'.format(toc(tt)))

if show_H:
    plt.figure()
    panplot = plt.imshow(img_pan)
    plt.title('Great Panorama')
    plt.show()


## Student Files
# first run "create_matching_points.py" with your own images to create a mat file with the matching coordinates.
max_err = 25  # <<<<< YOU MAY CHANGE THIS
inliers_percent = 0.8  # <<<<< YOU MAY CHANGE THIS

img_src_test = mpimg.imread('src_test.jpg')
img_dst_test = mpimg.imread('dst_test.jpg')

matches_test = scipy.io.loadmat('matches_test')

match_p_dst = matches_test['match_p_dst']
match_p_src = matches_test['match_p_src']

# Display the matching points on both images and check if they are indeed a perfect match
display_matching_test(src_image=img_src_test, dest_image=img_dst_test, matching=create_matching_dict(matches_test))

# Build student panorama

tt = tic()
img_pan = panorama(img_src_test, img_dst_test, match_p_src, match_p_dst, inliers_percent, max_err)
print('Student Panorama {:5.4f} sec'.format(toc(tt)))

plt.figure()
panplot = plt.imshow(img_pan)
plt.title('Awesome Panorama')
plt.show()
