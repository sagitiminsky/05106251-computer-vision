import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import svd,lstsq
import random
########################################
#### Part A: Homography computation ####
########################################


def create_matching_dict(matching):
    """
    Creates a dict with keys {src,dest} where each value is a 2xN array representing the matching points
    :param matching:
    :return: dict
    """

    return {
        'src': matching['match_p_src'].astype(float),
        'dest': matching['match_p_dst'].astype(float)

    }


def display_matching(src_image, dest_image, perfect_matching, matching):
    """
    Display the matching points on both images and check if they are indeed a perfect match.
    :param src_image: The source image in matrix representaion
    :param dst_image: The dest image in matrix representation
    :param perfect_matching: dict with keys {src, dest} each is a 2xN array representing the matching points in the src image
    :param matching: dict with keys {src, dest} each is a 2xN array representing the matching points in the src image
    :return: None
    """

    plt.close('all')
    f, axarr = plt.subplots(2, 2)

    # src
    axarr[0, 0].imshow(src_image)
    axarr[0, 0].set_title('perfect matching src')
    axarr[0, 0].scatter(x=perfect_matching['src'][0], y=perfect_matching['src'][1], c='r', s=16)

    axarr[1, 0].set_title('matching src')
    axarr[1, 0].imshow(src_image)
    axarr[1, 0].scatter(x=matching['src'][0], y=matching['src'][1], c='g', s=16)

    # dest
    axarr[0, 1].imshow(dest_image)
    axarr[0, 1].set_title('perfect matching dst')
    axarr[0, 1].scatter(x=perfect_matching['dest'][0], y=perfect_matching['dest'][1], c='r', s=16)

    axarr[1, 1].imshow(dest_image)
    axarr[1, 1].set_title('matching dst')
    axarr[1, 1].scatter(x=matching['dest'][0], y=matching['dest'][1], c='g', s=16)

    plt.show()


#########################################
##### Part B: Dealing with outliers #####
#########################################

def add_ones(points1,points2):
    points1 = points1.T
    points2 = points2.T
    if points1.shape[0] != points2.shape[0]: raise ValueError("The number of input and output points mismatches")
    if points1.shape[1] == 2:
        p1 = np.ones((len(points1), 3), 'float64')
        p1[:, :2] = points1
    elif points1.shape[1] == 3:
        p1 = points1
    else:
        raise ValueError("Bad shape for input points")

    if points2.shape[1] == 2:
        p2 = np.ones((len(points2), 3), 'float64')
        p2[:, :2] = points2
    elif points2.shape[1] == 3:
        p2 = points2
    else:
        raise ValueError("Bad shape for output points")

    npoints = len(points1)
    return p1,p2,npoints

def compute_homography_naive(points1, points2):

    p1,p2,npoints=add_ones(points1,points2)

    A = np.zeros((3 * npoints, 9), 'float64')

    for i in range(npoints):
        p1i = p1[i]
        x2i, y2i, w2i = p2[i]
        xpi = x2i * p1i
        ypi = y2i * p1i
        wpi = w2i * p1i

        A[i * 3, 3:6] = -wpi
        A[i * 3, 6:9] = ypi
        A[i * 3 + 1, 0:3] = wpi
        A[i * 3 + 1, 6:9] = -xpi
        A[i * 3 + 2, 0:3] = -ypi
        A[i * 3 + 2, 3:6] = xpi

    U, s, Vt = svd(A, full_matrices=False, overwrite_a=True)
    del U, s
    h = Vt[-1]
    H = h.reshape(3, 3)
    return H

def p_transformed_cartesian(vec):
    tx=vec[0]
    ty=vec[1]
    tz=vec[2]

    px = tx / tz
    py = ty / tz
    Z = 1 / tz

    return np.array([px,py,Z])

def test_homography(H, mp_src, mp_dst, max_err):
    p1, p2,npoints = add_ones(mp_src, mp_dst)
    ninliers,noutliers,squared_error=0,0,0

    for src_p,dest_p in zip(p1,p2):
        p_squared_error=sum((p_transformed_cartesian(H @ src_p)[:2] - dest_p[:2]) ** 2) ** 0.5
        if p_squared_error<=max_err:
           squared_error+=p_squared_error
           ninliers+=1
        else:
            noutliers+=1


    if ninliers+noutliers!=npoints:
        raise ValueError("ninliers+noutliers!=npoints")

    if ninliers==0:
        print("number of outliers:{}".format(noutliers))
        raise ValueError("number of inliers is zero")


    fit_percent= ninliers/npoints
    dist_mse=squared_error/ninliers
    return fit_percent,dist_mse


def compute_homography(match_p_src, match_p_dst, inliers_percent, max_err):
    final_H=None
    final_fit_percent,final_dist_mse=0.0,None
    for i in range(1000):
        #find 4 random points
        random_four=random.sample(range(len(match_p_src.T)), 4)

        H=compute_homography_naive(match_p_src.T[random_four].T,match_p_dst.T[random_four].T)
        fit_percent, dist_mse=test_homography(H, match_p_src, match_p_dst, max_err)

        if final_fit_percent<fit_percent:
            final_fit_percent=fit_percent
            final_dist_mse=dist_mse
            final_H=H

        if fit_percent>inliers_percent:
            break

    return final_H




#########################################
####### Part C: Panorama creation #######
#########################################





if __name__ == '__main__':
    raise NotImplementedError('ex1_functions.py is only callable from test_script.py')
