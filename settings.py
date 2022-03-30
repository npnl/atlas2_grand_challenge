from isles.scoring import dice_coef, volume_difference, simple_lesion_count_difference, precision, sensitivity, \
  specificity, accuracy, lesion_count_by_weighted_assignment

eval_settings = {
    "GroundTruthRoot": "/opt/evaluation/ground-truth/",
    "PredictionRoot": "/input/",
    "GroundTruthBIDSDerivativeName": ["atlas2"],
    "PredictionBIDSDerivativeName": ["atlas2_prediction"],
    "GroundTruthEntities": {
        "subject": "",
        "session": "",
        "suffix": "mask"
    },
    "PredictionEntities": {
        "suffix": "mask"
    },
    "LoaderBatchSize": 4,
    "Multiprocessing": 8,
    "Aggregates": ["mean", "std", "min", "max", "25%", "50%", "75%", "count", "uniq", "freq"],
    "MetricsOutputPath": "/output/metrics.json",
    "SampleBIDS": "/opt/evaluation/sample_bids/",
    "ScoringFunctions": {'Dice': dice_coef,
                         'Volume Difference': volume_difference,
                         'Simple Lesion Count': simple_lesion_count_difference,
                         'Precision': precision,
                         'Sensitivity': sensitivity,
                         'Specificity': specificity,
                         'Accuracy': accuracy}
}
