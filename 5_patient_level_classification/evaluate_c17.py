"""
File:           evaluate_c17.py
Author:         Pavlina Koutecka
Date:           10/06/2020
Description:    This file makes the final prediction from features extracted from the tumor probability maps
                of every patient according to the task determined by the CAMELYON17 challenge
                (https://camelyon17.grand-challenge.org/Evaluation/) and stores the results to the final .csv file.

                Features of one slide are taken, classifier (Random Forest or XGBoost) makes his prediction
                on the slide (if the slide contains ITC/micrometastasis/macrometastasis or is negative),
                and after all the slides of one patient are evaluated, the final pN-stage prediction is made
                according to the official rules. Results of all the patients are stored to the final .csv file,
                which is submitted to the official CAMELYON17 submission web page.
"""


import numpy as np
import random
import pandas as pd
import pickle
from datetime import datetime
now = datetime.now()
import torch.nn.functional

import sys
import pathlib
parent_path = str(pathlib.Path().absolute().parent)
sys.path.append(parent_path)
sys.path.append(parent_path + '/2_preprocessing_and_visualization')
sys.path.append(parent_path + '/3_patch_level_segmentation')
import configuration as cfg
import utils

# random seed initialization to reproduce same results
manualSeed = 1
np.random.seed(manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
# random seed initialization in the case of using GPU
torch.cuda.manual_seed(manualSeed)
torch.cuda.manual_seed_all(manualSeed)
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def make_prediction(predictions):
    """
    This function determines the final pN-stage of the patient according to the official rules
    (https://camelyon17.grand-challenge.org/Evaluation/). It takes array with slide predictions of one
    patient, and using that, it determines the pN-stage.

    :param predictions: array with slide-level predictions (negative/ITC/micro/macro) of the patient
    :return: predicted pN-stage of the patient
    """

    # count number of slides predicted as negative (0), ITC (1), micrometastasis (2), macrometastasis (3)
    count_0, count_1, count_2, count_3 = predictions.count(0), predictions.count(1), predictions.count(2), predictions.count(3)

    # predict the patient pN-stage according to the official rules
    if count_0 == 5:  # no tumor cells
        stage = 'pN0'
    elif count_2 == 0 and count_3 == 0:  # only ITCs
        stage = 'pN0(i+)'
    elif count_3 == 0:  # only micrometastases
        stage = 'pN1mi'
    elif count_3 < 4:  # metastases in 1-3 nodes, at least 1 macro
        stage = 'pN1'
    else:  # metastases in 4-9 nodes, at least 1 macro
        stage = 'pN2'

    return stage


def predict_stage():
    """
    This function makes the final prediction from features extracted from the tumor probability maps
    of every patient according to the task determined by the CAMELYON17 challenge and stores the
    results to the final .csv file.
    First, the function loads the submission .csv template file with patient_ids. Also
    the classifier needed for evaluating every slide (negative/ITC/micro/macro) is loaded. We can
    choose from two trained classifiers - Random Forest and XGBoost. After that, the process of
    making the slides and patients predictions starts. Features of the slide are taken, classifier makes
    his prediction on the slide, and after all the slides of one patient are evaluated, the final
    pN-stage prediction is made according to the official rules.
    Results of all the patients are stored to the final .csv file, which is submitted to the official
    CAMELYON17 submission web page.

    :return: None
    """

    # define all the possible tumor types appearing on the slides
    stages = {0: 'negative', 1: 'itc', 2: 'micro', 3: 'macro'}

    # load the .csv template with prepared patients
    submission_df = pd.read_csv(evaluate_folder + csv_file)
    submission_map = {df_row[0]: df_row[1].lower() for _, df_row in submission_df.iterrows() if df_row[0].lower().endswith('.tif')}

    # load the classifier predicting the tumor type
    loaded_classifier = pickle.load(open(evaluate_folder + classifier_file, 'rb'))

    # prepare arrays for final writing into the submitting .csv file
    # (one for patient names and one for their final pN-stage predictions and tumor type predictions)
    final_names = []
    final_stages = []

    # prepare the first patient and array to store tumor type predictions of all his slides
    if args.data == 'c17_train':
        current_patient = 'patient_000'
    if args.data == 'c17_test':
        current_patient = 'patient_100'
    current_patient_predictions = []

    # run the cycle for all the patients and their slides
    for patient_id, reference_stage in submission_map.items():

        print(f"Predicting patient {patient_id}", end='\r')

        # check if the patient hasn't changed, if so, store predictions for him and move to the next patient
        if patient_id[:11] != current_patient:

            # make the final patient pN-stage prediction
            predicted_stage = make_prediction(current_patient_predictions)

            # store patient .zip name and pN-stage prediction to the final arrays
            final_names.append(current_patient + '.zip')
            final_stages.append(predicted_stage)

            # store patient .tif names and tumor type predictions to the final array
            for i in range(5):
                final_names.append(current_patient + f'_node_{i}.tif')
                final_stages.append(stages[current_patient_predictions[i]])

            # change the name of current patient and clear all the tumor type predictions
            current_patient = patient_id[:11]
            current_patient_predictions = []

        # load generated features of actual patients slide, make the tumor type prediction and store it to the array
        features = np.load(evaluate_folder + 'features/features_' + patient_id[:-4] + '.npy')
        slide_prediction = loaded_classifier.predict(features[np.newaxis, ...])
        current_patient_predictions.append(slide_prediction[0])

    # make all the computations also for the last slide of the last patient
    predicted_stage = make_prediction(current_patient_predictions)
    final_names.append(current_patient + '.zip')
    final_stages.append(predicted_stage)
    for i in range(5):
        final_names.append(current_patient + f'_node_{i}.tif')
        final_stages.append(stages[current_patient_predictions[i]])

    # store the final arrays to .csv file
    data = {'patient': final_names, 'stage': final_stages}
    df = pd.DataFrame(data)
    df.to_csv(evaluate_folder + 'csv_files/final.csv', index=False)


if __name__ == "__main__":

    args = utils.parse_args()

    print("Evaluating slides with regard to CAMELYON17 and creating .csv file...")

    # load correct .csv file
    if args.data == 'c17_train':
        csv_file = 'csv_files/stage_labels.csv'
    if args.data == 'c17_test':
        csv_file = 'csv_files/submission_example.csv'

    # load correct classifier
    if args.type == 'rf':
        classifier_file = 'classifiers/random_forest_classifier.sav'
    if args.type == 'xgb':
        classifier_file = 'classifiers/xgboost_classifier.sav'

    # prepare the evaluation folder - in this case, we are evaluating patients, not slides
    evaluate_folder = cfg.path.evaluate_patient

    # make all the tumor types and pN-stages predictions and store them to .csv file
    predict_stage()

    print("Process of evaluating slides with regard to CAMELYON17 and creating .csv file is done!")
