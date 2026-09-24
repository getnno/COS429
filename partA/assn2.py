import numpy as np
import cv2


def load_average_color_with_bias(X_data):
    """Compute one simple color feature vector per image.

    For each image, average over the height and width dimensions so that the
    red, green, and blue channels become three scalar features. The fourth
    feature is a constant bias term equal to 1.

    Arguments:
        X_data: numpy array of size (N, H, W, 3), where N is the number of
            images and the final dimension stores three color channels.

    Outputs:
        output: numpy array of size (N, 4). Columns 0-2 hold average color
            values, and column 3 holds the bias term.
    """
    X_data = X_data.copy()
    N = X_data.shape[0]
    output = np.zeros([N, 4], dtype=X_data.dtype)
    ### START YOUR CODE HERE ###
    # Fill output[:, :3] with one average RGB vector per image.
    # Fill output[:, 3] with the constant bias feature.
    for index, image in enumerate(X_data):
        sum = 0
        y_dim = image.shape[0]
        x_dim = image.shape[1]
        count = x_dim * y_dim
        for y in range(y_dim):
            for x in range(x_dim):
                sum += image[y][x]
        val = sum / count
        output[index] = np.append(val, 1)
        #output[index][0] = np.mean(image, where = )
    ### END YOUR CODE HERE ###

    return output


def load_flatten(X_data):
    """Flatten a batch of per-pixel or per-keypoint features.

    The input groups features by image. The output removes the image grouping
    and returns one row per feature vector.

    Arguments:
        X_data: numpy array of size (N, H * W, D)

    Outputs:
        output: numpy array of size (N * H * W, D)
    """
    X_data = X_data.copy()
    N, HW, D = X_data.shape
    X_data = X_data.copy()
    ### START YOUR CODE HERE ###
    # Reshape without changing feature values. The output should contain
    # N * H * W rows and D columns.
    output = np.zeros([N * HW, 128], dtype=X_data.dtype)
    index = 0
    for i in range(N):
        for j in range(HW):
            output[index] = X_data[i][j]
            index+=1
    # print(output.shape)
    ### END YOUR CODE HERE ###

    return output


def load_histogram_with_bias(X_data, centroids):
    """Convert local descriptors into a bag-of-visual-words histogram.

    Each descriptor is assigned to its nearest centroid. Each image then gets
    a histogram counting how many descriptors chose each centroid. The final
    column is a constant bias term equal to 1.

    Arguments:
        X_data: numpy array of size (N, P, D), where N is number of images,
                P is number of keypoints, and D is dimension of features
        centroids: numpy of array of size (K, D), where K is number of centroids.

    Outputs:
        X_hist: numpy array of size (N, K+1), where X_hist[i,j] contains number of
                keypoints from image i that is closest to centroid[j].
                X_hist[:, K] should be 1 for bias.
    """
    X_data, centroids = X_data.copy(), centroids.copy()
    N, P, D = X_data.shape
    K, D = centroids.shape
    X_hist = np.zeros([N, K + 1])

    ### START YOUR CODE HERE ###
    # For every image and descriptor, compute distances to all K centroids.
    # Increment the bin for the nearest centroid, then set the last column
    # of each image row to the bias value.
    temp = np.zeros([K])
    for i in range(N):
        for j in range(P):
            for k in range(K):
                temp[k] = np.linalg.norm(X_data[i][j]-centroids[k])
            X_hist[i][np.argmin(temp)] += 1
        X_hist[i][K] = 1
    ### END YOUR CODE HERE ###

    return X_hist


def load_vector_image_with_bias(X_train, X_val, X_test):
    """Create normalized raw-pixel feature rows for train, validation, and test.

       Reshape each image into one long vector, subtract the mean training
       image from every split, and append one constant bias feature. Use only
       the training split to compute the mean so validation and test data do
       not leak into preprocessing.

    Arguments:
        X_train: numpy array of size (N_train, H, W, 3), where N_train is number of images
        X_val: numpy array of size (N_val, H, W, 3), where N_val is number of images
        X_test: numpy array of size (N_test, H, W, 3), where N_test is number of images

    Outputs:
        X_train: numpy array of size (N, H * W * 3 + 1). Bias dimension at the end.
        X_val: numpy array of size (N, H * W * 3 + 1). Bias dimension at the end.
        X_test: numpy array of size (N, H * W * 3 + 1). Bias dimension at the end.
    """
    X_train, X_val, X_test = X_train.copy(), X_val.copy(), X_test.copy()
    N_train, N_val, N_test = X_train.shape[0], X_val.shape[0], X_test.shape[0]

    ### START YOUR CODE HERE ###
    # Flatten each split to two dimensions: one row per image.
    # Compute the mean image vector from flattened X_train only.
    # Subtract that mean from train, validation, and test features.
    # Append one bias column of ones to each split.
    X_train = np.reshape(X_train, (N_train,-1))
    X_val = np.reshape(X_val, (N_val, -1))
    X_test = np.reshape(X_test, (N_test, -1))

    mean = np.mean(X_train, axis = 0)
    X_train = X_train - mean
    X_val = X_val - mean
    X_test = X_test - mean
    for i in range(N_train):
        np.append(X_train[i],1)
    for i in range(N_val):
        np.append(X_val[i],1)
    for i in range(N_test):
        np.append(X_test[i],1)
    ### END YOUR CODE HERE ###

    return X_train, X_val, X_test
