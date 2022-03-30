from isles.scoring import dice_coef, volume_difference, simple_lesion_count_difference, precision, sensitivity, \
  specificity, accuracy, lesion_count_by_weighted_assignment

eval_settings = {
    "GroundTruthRoot": "/opt/evaluation/ground-truth/",     # Path to the ground truth
    "PredictionRoot": "/input/",                            # Path to the user predictions
    "GroundTruthBIDSDerivativeName": ["atlas2"],            # BIDS derivative name of the ground truth
    "PredictionBIDSDerivativeName": ["atlas2_prediction"],  # BIDS derivative name of the predictions
    "GroundTruthEntities": {                                # BIDS entities identifying the ground truth
        "subject": "",
        "session": "",
        "suffix": "mask"
    },
    "PredictionEntities": {                                 # BIDS entities identifying the predictions
        "suffix": "mask"
    },
    "LoaderBatchSize": 4,                                   # Number of images to load at a time
    "Multiprocessing": 8,                                   # Number of processors to use in parallel
    "Aggregates": ["mean", "std", "min", "max", "25%", "50%", "75%", "count", "uniq", "freq"],  # Summary stats to use
    "MetricsOutputPath": "/output/metrics.json",            # Desired location of output summary
    "SampleBIDS": "/opt/evaluation/sample_bids/",           # Path to the sample BIDS directory; don't modify.
    "ScoringFunctions": {'Dice': dice_coef,                 # Functions to use for scoring the dataset.
                         'Volume Difference': volume_difference,
                         'Simple Lesion Count': simple_lesion_count_difference,
                         'Precision': precision,
                         'Sensitivity': sensitivity,
                         'Specificity': specificity,
                         'Accuracy': accuracy}
}
