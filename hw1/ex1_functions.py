import matplotlib.pyplot as plt


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


if __name__ == '__main__':
    raise NotImplementedError('ex1_functions.py is only callable from test_script.py')