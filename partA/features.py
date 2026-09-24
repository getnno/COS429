import numpy as np
import cv2
import matplotlib.pyplot as plt


############## SIFT ######################################################

def extract_sift(img, step_size=1):
    """
    Extract dense SIFT features for a grayscale image.

    Instead of asking OpenCV to detect interest points, this assignment places
    keypoints on a regular grid. Every `step_size` pixels in both the vertical
    and horizontal directions, create a `cv2.KeyPoint` and compute its SIFT
    descriptor. This makes the output deterministic and easier to grade.

    Note: Check sift.compute and cv2.KeyPoint

    Args:
        img: Grayscale image of shape (H, W)
        step_size: Size of the step between keypoints.

    Return:
        descriptors: numpy array of shape (int(img.shape[0]/step_size) * int(img.shape[1]/step_size), 128)
                     contains sift feature.
    """
    sift = cv2.SIFT_create()  # or cv2.xfeatures2d.SIFT_create()
    descriptors = np.zeros(
        (int(img.shape[0] / step_size) * int(img.shape[1] / step_size), 128))

    ### START YOUR CODE HERE ###
    # Build keypoints in a consistent grid order, then pass that list to
    # sift.compute. Store only the descriptor matrix in `descriptors`.
    xDim = img.shape[1]
    yDim = img.shape[0]
    kp = []
    for x in range(0,xDim,step_size):
        for y in range(0,yDim, step_size):
            kp.append(cv2.KeyPoint(x, y , step_size))
    keypoints, descriptors = sift.compute(img, kp)
    ### END YOUR CODE HERE ###

    return descriptors


def extract_sift_for_dataset(data, step_size=1):
    """Run dense SIFT extraction for every image in a dataset.

    `data` is expected to contain color images. Each image is converted to
    grayscale before calling `extract_sift`, because SIFT descriptors operate
    on single-channel intensity images.
    """
    all_features = []
    for i in range(len(data)):
        img = data[i]
        img = cv2.cvtColor(np.uint8(img), cv2.COLOR_BGR2GRAY)
        descriptors = extract_sift(img, step_size)
        all_features.append(descriptors)
    return np.stack(all_features, 0)
