import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import svd,lstsq

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


def compute_homography_naive(points1, points2):
    points1=points1.T
    points2=points2.T
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



def test_homography(H, mp_src, mp_dst, max_err):
    for i



if __name__ == '__main__':
    raise NotImplementedError('ex1_functions.py is only callable from test_script.py')
