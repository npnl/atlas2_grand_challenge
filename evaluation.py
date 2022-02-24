from evalutils import ClassificationEvaluation
from stroke.bids_loader import BIDSLoader
# from .scoring import dice_coef, volume_difference, simple_lesion_count_difference, lesion_count_by_weighted_assignment
import pandas as pd
import json
from os.path import join, dirname
from collections import defaultdict
import numpy as np
import scipy.ndimage
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import precision_score
config_json_path = join(dirname(__file__), 'config.json')



class Atlas2(ClassificationEvaluation):
    def __init__(self,
                 prediction_root: str = None,
                 ground_truth_root: str = None,
                 aggregates: set = None
                 ):
        self.config = json.load(open(config_json_path, 'r'))
        if(prediction_root is None):
            self.prediction_root = self.config['PredictionRoot']
        else:
            self.prediction_root = prediction_root
        if(ground_truth_root is None):
            self.ground_truth_root = self.config['GroundTruthRoot']
        else:
            self.ground_truth_root = ground_truth_root

        if(aggregates is None):
            self.aggregates = {'mean', 'std', 'min', 'max', '25%', '50%', '75%', 'count', 'uniq', 'freq'}
        else:
            self.aggregates = aggregates



        self.score_functions = {'Sørensen–Dice': dice_coef,
                                'Volume Difference': volume_difference,
                                'Simple Lesion Count': simple_lesion_count_difference,
                                'LCWA': lesion_count_by_weighted_assignment}
        self.loader = None  # Defined in _prepare_data_list
        self.score_lists = defaultdict(list)  # Used in evaluate(); dict of scores found for samples

        # Prepare data list
        self._prepare_data_list()
        return

    def evaluate(self):
        # Load data iteratively
        for prediction, truth in self.loader.load_batches():
            # Score the data
            scores = self.score(prediction, truth)
            for name, score_values in scores.items():
                self.score_lists[name] += score_values

        # Compute mean for each score across samples
        # score_means = defaultdict(float)
        # for name, score in self.score_lists:
        #     score_means[name] = np.mean(score)
        score_summary = self.aggregate_scores(self.score_lists, self.aggregates)
        # Save score
        self.save_score(score_summary)

        return

    @staticmethod
    def aggregate_scores(scores, aggregates):
        '''
        Returns the aggregate measures in scores.
        Parameters
        ----------
        scores : dict
            Dictionaries of score_name:sample_scores to aggregate.
        aggregates : list
            List of strings containing valid keys of pandas.Series.describe().

        Returns
        -------

        '''
        score_des = pd.DataFrame(scores).describe()
        score_summary = defaultdict(dict)
        for score_name in score_des.keys():
            for score_agg in score_des[score_name].keys():
                if(score_agg in aggregates):
                    score_summary[score_name][score_agg] = score_des[score_name][score_agg]
        return score_summary

    @staticmethod
    def save_score(scores: dict,
                   output_path: str = '/output/metrics.json'):
        '''
        Saves the score summary.
        Parameters
        ----------
        scores : dict
            Dictionary summarizing score results.
        output_path: str
            Path where to save the data.

        Returns
        -------
        None
        '''
        f = open(output_path, 'w')
        json.dump(scores, f)
        f.close()
        return

    def _prepare_data_list(self):
        loader_kwargs = {'target_entities': self.config['GroundTruthEntities'],
                         'data_entities': self.config['PredictionEntities'],
                         'root_list': [self.config['PredictionRoot'],
                                       self.config['GroundTruthRoot']],
                         'batch_size': self.config['LoaderBatchSize']}

        if(len(self.config['PredictionBIDSDerivativeName']) > 0):
            loader_kwargs['data_derivatives_names'] = self.config['PredictionBIDSDerivativeName']
        if(len(self.config['GroundTruthBIDSDerivativeName']) > 0):
            loader_kwargs['target_derivatives_names'] = self.config['GroundTruthBIDSDerivativeName']
        self.loader = BIDSLoader(**loader_kwargs)
        return

    def score(self, prediction, truth):
        '''

        Returns
        -------

        '''
        score_dict = {}
        for name, score_function in self.score_functions.items():
            score_dict[name] = score_function(prediction=prediction, truth=truth, batchwise=True)
        return score_dict


import numpy as np
import scipy.ndimage
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import precision_score

def dice_coef(prediction, truth, batchwise=False):
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

    # Compute dice coef
    coef_list = []
    for i in range(prediction.shape[0]):
        coef = prediction[i:i+1, ...] @ truth[i:i+1, ...].T
        coef_list.append(coef)

    # Return list of coeffs if batchwise, otherwise return float
    if(batchwise):
        return tuple(coef_list)
    else:
        return coef_list[0]


def volume_difference(prediction, truth, batchwise=False):
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


def simple_lesion_count_difference(prediction, truth, batchwise=False):
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
        pred_shape = prediction.shape
        truth_shape = truth.shape
        count_list = []
        for i in range(pred_shape[0]):
            _, pred_count = scipy.ndimage.label(prediction[i, ...])
            _, truth_count = scipy.ndimage.label(truth[i, ...])
            count_list.append(np.abs(pred_count - truth_count))
        return tuple(count_list)
    return


def lesion_count_by_weighted_assignment(prediction, truth, batchwise=False):
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
    for idx_sample in range(pred_shape[0]):
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


if __name__ == "__main__":
    Atlas2().evaluate()
