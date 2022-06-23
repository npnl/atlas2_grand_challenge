import numpy as np
import scipy.ndimage
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import precision_score, recall_score, accuracy_score
import cc3d

def dice_coef(truth, prediction, batchwise=False):
    '''
    Computes the Sørensen–Dice coefficient for the input matrices. If batchwise=True, the first dimension of the input
    images is assumed to indicate the batch, and this function will return a coefficient for each sample. i.e., images
    of dimension (4,1,20,20,20) would return 4 coefficients.
    Parameters
    ----------
    prediction : np.array
        Array containing the prediction.
    truth : np.array
        Array containing the ground truth.
    batchwise : bool
        Optional. Indicate whether the computation should be done batchwise, assuming that the first dimension of the
        data is the batch. Default: False.
    Returns
    -------
    float or tuple
        Sørensen–Dice coefficient.
    '''

    # Reshape the input to reduce computation to a dot product
    if(not batchwise):
        prediction = np.reshape(prediction, (1,np.prod(prediction.shape)))
        truth = np.reshape(truth, (1,np.prod(truth.shape)))
    else:
        pred_shape = prediction.shape
        prediction = np.reshape(prediction, (pred_shape[0], np.prod(pred_shape[1:])))
        truth_shape = truth.shape
        truth = np.reshape(truth, (truth_shape[0], np.prod(truth_shape[1:])))

    # Prevent values >1 from inflating the score
    np.clip(truth, 0, 1, out=truth)
    np.clip(prediction, 0, 1, out=prediction)

    # Compute dice coef
    coef_list = []
    for i in range(truth.shape[0]):
        coef_denom = np.sum(prediction[i,...]) + np.sum(truth[i,...])
        if(coef_denom == 0):  # If there are no non-zero labels in either the truth or the prediction
            coef_list.append(1.0)  # "Perfect" score
            continue
        coef = prediction[i:i+1, ...] @ truth[i:i+1, ...].T
        coef = 2*coef / coef_denom
        coef_list.append(float(coef))

    # Return list of coeffs if batchwise, otherwise return float
    if(batchwise):
        return tuple(coef_list)
    else:
        return coef_list[0]


def volume_difference(truth, prediction, batchwise=False):
    '''
    Computes the total volume difference between the prediction and ground truth.
    Parameters
    ----------
    prediction : np.array
        Array containing the prediction.
    truth : np.array
        Array containing the ground truth.
    batchwise : bool
        Optional. Indicate whether the computation should be done batchwise, assuming that the first dimension of the
        data is the batch. Default: False.
    Returns
    -------
    float or tuple
    '''
    if(not batchwise):
        pred_vol = np.sum(prediction)
        truth_vol = np.sum(prediction)
        return np.abs(pred_vol - truth_vol)
    else:
        pred_shape = prediction.shape
        prediction = np.reshape(prediction, (pred_shape[0], np.prod(pred_shape[1:])))
        truth_shape = truth.shape
        truth = np.reshape(truth, (truth_shape[0], np.prod(truth_shape[1:])))

        prediction_vols = [np.sum(prediction[i, ...]) for i in range(pred_shape[0])]
        truth_vols = [np.sum(truth[i,...]) for i in range(truth_shape[0])]
        return tuple(np.abs(pred - tru) for pred, tru in zip(prediction_vols, truth_vols))
    return


def simple_lesion_count_difference(truth, prediction, batchwise=False):
    '''
    Computes the difference in the number of distinct regions between the two input images. Regions are considered
    distinct if there exists no path between two area that is entirely inside the region. Note that no evaluation
    of the regions is done; the regions in one image could be entirely non-overlapping with the regions of the other
    image, but this function would return '0' if the number of regions match.
    Parameters
    ----------
    prediction : np.array
        Array containing the prediction.
    truth : np.array
        Array containing the ground truth.
    batchwise : bool
        Optional. Indicate whether the computation should be done batchwise, assuming that the first dimension of the
        data is the batch. Default: False.

    Returns
    -------
    int or tuple
    '''
    if(not batchwise):
        _, pred_count = scipy.ndimage.label(prediction)
        _, truth_count = scipy.ndimage.label(truth)
        return np.abs(pred_count - truth_count)
    else:
        truth_shape = truth.shape
        count_list = []
        for i in range(truth_shape[0]):
            _, pred_count = scipy.ndimage.label(prediction[i, ...])
            _, truth_count = scipy.ndimage.label(truth[i, ...])
            count_list.append(np.abs(pred_count - truth_count))
        return tuple(count_list)
    return


def lesion_count_by_weighted_assignment(truth, prediction, batchwise=False):
    '''
    Performs lesion matching between the predicted lesions and the true lesions. A weighted bipartite graph between
    the predicted and true lesions is constructed, using precision as the edge weights. The returned value is the
    mean precision across predictions normalized by the number of lesions in the ground truth. Values close to 1
    indicate that the right number of lesions have been identified and that they overlap. Lower values indicate either
    the wrong number of predicted lesions or that they do not sufficiently overlap with the ground truth.
    Parameters
    ----------
    prediction : np.array
        Array containing the prediction.
    truth : np.array
        Array containing the ground truth.
    batchwise : bool
        Optional. Indicate whether the computation should be done batchwise, assuming that the first dimension of the
        data is the batch. Default: False.

    Returns
    -------
    float or tuple
    '''
    # Reshape to avoid code duplication
    if(not batchwise):
        prediction = np.reshape(prediction, (1, *prediction.shape))
        truth = np.reshape(truth, (1, *truth.shape))
    pred_shape = prediction.shape
    truth_shape = truth.shape

    # "Lesion Count by Weighted Assignment"
    lcwa = []
    for idx_sample in range(truth_shape[0]):
        # Identify unique regions
        pred_lesions, num_pred_lesions = scipy.ndimage.label(prediction[idx_sample, ...])
        truth_lesion, num_truth_lesions = scipy.ndimage.label(truth[idx_sample, ...])

        # reshape for use with sklearn precision
        pred_reshape = np.reshape(prediction[idx_sample, ...], (np.prod(pred_shape[1:])))
        truth_reshape = np.reshape(truth[idx_sample, ...], (np.prod(truth_shape[1:])))

        # pre-allocate cost matrix
        cost_matrix = np.zeros((num_pred_lesions, num_truth_lesions))

        # compute cost matrix
        for idx_pred in range(num_pred_lesions):
            pred_lesion = pred_reshape == idx_pred
            for idx_truth in range(num_truth_lesions):
                truth_lesion = truth_reshape == idx_truth
                cost_matrix[idx_pred, idx_truth] = precision_score(y_true=truth_lesion, y_pred=pred_lesion)
        row_ind, col_ind = linear_sum_assignment(cost_matrix=cost_matrix, maximize=True)
        total_precision = cost_matrix[row_ind, col_ind].sum()
        lcwa.append(total_precision / num_truth_lesions)

    if(not batchwise):
        return lcwa[0]
    else:
        return tuple(lcwa)


def precision(truth, prediction, batchwise=False):
    '''
    Returns the precision of the prediction: tp / (tp + fp)
    Parameters
    ----------
    truth : np.array
        Ground truth data.
    prediction : np.array
        Prediction data.
    batchwise : bool
        Optional. Indicate whether the computation should be done batchwise, assuming that the first dimension of the
        data is the batch. Default: False.
    Returns
    -------
    float or tuple
        Precision of the input. If batchwise=True, the tuple is the precision for every sample.
    '''
    # sklearn implementation requires vectors of ints
    truth = np.round(truth).astype(np.uint8)
    prediction = np.round(prediction).astype(np.uint8)

    if(not batchwise):
        # Convert to vector
        num_pred = np.prod(prediction.shape)
        return precision_score(truth.reshape((num_pred,)), prediction.reshape((num_pred,)))
    else:
        # Need to get the precision for each sample in the batch
        precision_list = []
        num_pred = np.prod(prediction.shape[1:])
        for sample_idx in range(truth.shape[0]):
            sample_precision = precision(truth[sample_idx, ...].reshape((num_pred,)),
                                         prediction[sample_idx, ...].reshape((num_pred,)),
                                         batchwise=False)
            precision_list.append(sample_precision)
        return tuple(precision_list)


def sensitivity(truth, prediction, batchwise=False):
    '''
    Returns the sensitivity of the prediction: tp / (tp + fn)
    Parameters
    ----------
    truth : np.array
        Ground truth data.
    prediction : np.array
        Prediction data.
    batchwise : bool
        Optional. Indicate whether the computation should be done batchwise, assuming that the first dimension of the
        data is the batch. Default: False.

    Returns
    -------
    float or tuple
        Sensitivity of the input. If batchwise=True, the tuple is the sensitivity for every sample.
    '''
    return _recall(truth, prediction, batchwise=batchwise, pos_label=1)


def specificity(truth, prediction, batchwise=False):
    '''
    Returns the specificity of the prediction: tn / (tn + fp)
    Parameters
    ----------
    truth : np.array
        Ground truth data.
    prediction : np.array
        Prediction data.
    batchwise : bool
        Optional. Indicate whether the computation should be done batchwise, assuming that the first dimension of the
        data is the batch. Default: False.

    Returns
    -------
    float or tuple
        Specificity of the input. If batchwise=True, the tuple is the specificity for every sample.
    '''
    return _recall(truth, prediction, batchwise=batchwise, pos_label=0)


def _recall(truth, prediction, batchwise=False, pos_label=1):
    '''
    Returns the recall of the prediction: tp / (tp + fn), where 'positive' is defined by pos_label.
    Parameters
    ----------
    truth : np.array
        Ground truth data.
    prediction : np.array
        Prediction data.
    batchwise : bool
        Optional. Indicate whether the computation should be done batchwise, assuming that the first dimension of the
        data is the batch. Default: False.
    pos_label : int
        Optional. Indicate which label is the positive case. Default: 1.

    Returns
    -------
    float or tuple
        Recall of the input for the label specified by pos_label. If batchwise=True, the tuple is the specificity for every sample.
    '''
    # sklearn implementation requires vectors of ints
    truth = np.round(truth).astype(np.uint8)
    prediction = np.round(prediction).astype(np.uint8)

    if(not batchwise):
        # Convert to vector
        num_pred = np.prod(prediction.shape)
        return recall_score(truth.reshape((num_pred,)), prediction.reshape((num_pred,)), pos_label=pos_label)
    else:
        # Need to get the precision for each sample in the batch
        recall_list = []
        num_pred = np.prod(prediction.shape[1:])
        for sample_idx in range(truth.shape[0]):
            sample_recall = _recall(truth[sample_idx, ...].reshape((num_pred,)),
                                    prediction[sample_idx, ...].reshape((num_pred,)),
                                    batchwise=False,
                                    pos_label=pos_label)
            recall_list.append(sample_recall)
        return tuple(recall_list)


def accuracy(truth, prediction, batchwise=False):
    '''
    Returns the accuracy of the prediction (tp + tn) / (tp+tn+fp+fn)
    Parameters
    ----------
    truth : np.array
        Ground truth data.
    prediction : np.array
        Prediction data.
    batchwise : bool
        Optional. Indicate whether the computation should be done batchwise, assuming that the first dimension of the
        data is the batch. Default: False.

    Returns
    -------
    float or tuple
        Accuracy of the input. If batchwise=True, the tuple is the specificity for every sample.
    '''
    # sklearn implementation requires vectors of ints
    truth = np.round(truth).astype(np.uint8)
    prediction = np.round(prediction).astype(np.uint8)

    if(not batchwise):
        # Convert to vector
        num_pred = np.prod(prediction.shape)
        return accuracy_score(truth.reshape((num_pred,)), prediction.reshape((num_pred,)))
    else:
        accuracy_list = []
        num_pred = np.prod(prediction.shape[1:])
        for sample_idx in range(truth.shape[0]):
            sample_accuracy = accuracy(truth[sample_idx, ...].reshape((num_pred,)),
                                       prediction[sample_idx,...].reshape((num_pred,)),
                                       batchwise=False)
            accuracy_list.append(sample_accuracy)
        return tuple(accuracy_list)


def compute_lesion_f1_score(truth, prediction, empty_value=1.0, connectivity=26):
    """
    Computes the lesion-wise F1-score between two masks.

    Parameters
    ----------
    ground_truth : array-like, bool
        Any array of arbitrary size. If not boolean, will be converted.
    prediction : array-like, bool
        Any other array of identical size as 'ground_truth'. If not boolean, it will be converted.
    empty_value : scalar, float.
    connectivity : scalar, int.

    Returns
    -------
    f1_score : float
        Lesion-wise F1-score as float.
        Max score = 1
        Min score = 0
        If both images are empty (tp + fp + fn =0) = empty_value

    Notes
    -----
    This function computes lesion-wise score by defining true positive lesions (tp), false positive lesions (fp) and
    false negative lesions (fn) using 3D connected-component-analysis.

    tp: 3D connected-component from the ground-truth image that overlaps at least on one voxel with the prediction image.
    fp: 3D connected-component from the prediction image that has no voxel overlapping with the ground-truth image.
    fn: 3d connected-component from the ground-truth image that has no voxel overlapping with the prediction image.
    """
    truth = np.asarray(truth).astype(bool)
    prediction = np.asarray(prediction).astype(bool)
    tp = 0
    fp = 0
    fn = 0

    # Check if ground-truth connected-components are detected or missed (tp and fn respectively).
    intersection = np.logical_and(truth, prediction)
    labeled_ground_truth, N = cc3d.connected_components(truth, connectivity=connectivity, return_N=True)

    # Iterate over ground_truth clusters to find tp and fn.
    # tp and fn are only computed if the ground-truth is not empty.
    if N > 0:
        for _, binary_cluster_image in cc3d.each(labeled_ground_truth, binary=True, in_place=True):
            if np.logical_and(binary_cluster_image, intersection).any():
                tp += 1
            else:
                fn += 1

    # iterate over prediction clusters to find fp.
    # fp are only computed if the prediction image is not empty.
    labeled_prediction, N = cc3d.connected_components(prediction, connectivity=connectivity, return_N=True)
    if N > 0:
        for _, binary_cluster_image in cc3d.each(labeled_prediction, binary=True, in_place=True):
            if not np.logical_and(binary_cluster_image, truth).any():
                fp += 1

    # Define case when both images are empty.
    if tp + fp + fn == 0:
        _, N = cc3d.connected_components(truth, connectivity=connectivity, return_N=True)
        if N == 0:
            f1_score = empty_value
    else:
        f1_score = tp / (tp + (fp + fn) / 2)

    return f1_score

def compute_absolute_lesion_count_difference(truth, prediction, connectivity=26):
    """
    Computes the absolute lesion difference between two masks. The number of lesions are counted for
    each volume, and their absolute difference is computed.
    Parameters
    ----------
    truth : array-like, bool
        Any array of arbitrary size. If not boolean, will be converted.
    prediction : array-like, bool
        Any other array of identical size as 'ground_truth'. If not boolean, it will be converted.
    Returns
    -------
    abs_les_diff : int
        Absolute lesion difference as integer.
        Maximum similarity = 0
        No similarity = inf
    Notes
    -----
    """
    truth = np.asarray(truth).astype(bool)
    prediction = np.asarray(prediction).astype(bool)

    _, ground_truth_numb_lesion = cc3d.connected_components(truth, connectivity=connectivity, return_N=True)
    _, prediction_numb_lesion = cc3d.connected_components(prediction, connectivity=connectivity, return_N=True)
    abs_les_diff = abs(ground_truth_numb_lesion - prediction_numb_lesion)

    return abs_les_diff

def lesion_f1_score(truth, prediction, batchwise=False):
    """ Wrapper of compute_lesion_f1_score function to work batchwise

    Parameters
    ----------
    truth : array-like, bool
        Any array of arbitrary size. If not boolean, will be converted.
    prediction : array-like, bool
    batchwise : bool
        Optional. Indicate whether the computation should be done batchwise, assuming that the first dimension of the
        data is the batch. Default: False.

    Returns
    -------
    float or tuple
        Lesion-wise F1-score. If batchwise=True, the tuple is the F1-score for every sample.
    """

    if not batchwise:
        return compute_lesion_f1_score(truth, prediction)
    else:
        f1_list = []
        truth_shape = truth.shape
        for i in range(truth_shape[0]):
            f1_list.append(compute_lesion_f1_score(truth[i], prediction[i]))
        return tuple(f1_list)


def absolute_lesion_count_difference(truth, prediction, batchwise=False):
    """ Wrapper of compute_absolute_lesion_count_difference function to work batchwise

    Parameters
    ----------
    truth : array-like, bool
        Any array of arbitrary size. If not boolean, will be converted.
    prediction : array-like, bool
    batchwise : bool
        Optional. Indicate whether the computation should be done batchwise, assuming that the first dimension of the
        data is the batch. Default: False.

    Returns
    -------
    int or tuple
        Absolute lesion count difference. If batchwise=True, the tuple is the Absolute lesion count difference
        for every sample.
    """

    if not batchwise:
        return compute_absolute_lesion_count_difference(truth, prediction)
    else:
        absolute_lesion_count_diff_list = []
        truth_shape = truth.shape
        for i in range(truth_shape[0]):
            absolute_lesion_count_diff_list.append(compute_absolute_lesion_count_difference(truth[i], prediction[i]))
        return tuple(absolute_lesion_count_diff_list)
