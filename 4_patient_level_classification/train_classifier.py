"""
File:           train_classifier.py
Author:         Pavlina Koutecka
Date:           25/05/2020
Description:    This file prepares and trains classifiers for the slide-level stage (negative/ITC/micrometastasis/
                macrometastasis) predictions of the WSIs tumor probability maps. Classifiers are trained on the
                features from the tumor probability maps generated in the generate_features.py file.

                Currently, there are two classifiers that can be trained - XGBoost and Random Forest. For each
                of them, you can find optimal parameters using grid search with K-fold cross validation (you
                should pass additional parameter --type to 'xgb_grid' or 'rf_grid'), train the classifier and evaluate
                it using K-fold cross validation (you should pass additional parameter --type to 'xgb_train' or
                'rf_train') or print stats of the K-fold cross validation (you should pass additional parameter
                --type to 'xgb_stats' or 'rf_stats').
"""


import numpy as np
import random
import pandas as pd
import pickle
import xgboost as xgb
import matplotlib.pyplot as plt
from datetime import datetime
now = datetime.now()

import torch.nn.functional
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import sys
import pathlib
parent_path = str(pathlib.Path().absolute().parent)
sys.path.append(parent_path)
sys.path.append(parent_path + '/2_preprocessing_and_visualization')
sys.path.append(parent_path + '/3_slide_level_classification')
import configuration as cfg
import utils
import plot_utils

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


def train_xgboost():
    """
    This function prepares the XGBoost classifier to predict the slide-level stage (negative/ITC/micrometastasis/
    macrometastasis) of the WSI. You can choose from three options:

        > XGBoost grid - this part performs the K-Fold (5-Fold) cross validation grid search to find optimal
        parameters for the XGBoost classifier training.
        > XGBoost train - this part trains the XGBoost classifier with optimal parameters, saves it to .sav
        file and creates additional visualizations. Performance of the classifier is evaluated by the K-Fold
        (5-Fold) cross validation to ensure unbiased results.
        > XGBoost stats - this parts prints evaluation stats of every fold evaluated during the K-Fold cross
        validation. It also creates and saves confusion matrices.

    :return: None

    Useful links:
        https://www.datacamp.com/community/tutorials/xgboost-in-python
        https://xgboost.readthedocs.io/en/latest/parameter.html
        https://scikit-learn.org/stable/modules/grid_search.html
    """

    # perform the K-Fold cross validation grid search to find best parameters of the XGBoost classifier
    if args.type == 'xgb_grid':

        print("Performing the K-Fold cross validation grid search for the XGBoost...")

        # prepare the XGBoost classifier and all the parameters that should be involved in the grid searching
        xgb_classifier = xgb.XGBClassifier()
        parameters = {
            "eta": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40],  # default=0.3, alias: learning_rate, range:[0,1]
            "max_depth": [3, 4, 5, 6, 8, 10, 15, 20],  # default=6, range:[0,inf]
            "gamma": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],  # default=0, alias: min_split_loss, range:[0,inf]
            "n_estimators": [100, 200, 300],  # default=100
            "subsample": [0.5, 0.6, 0.8, 1]  # default=1, range:(0,1]
        }

        # perform the K-Fold (5-Fold) cross validation grid search on the training dataset
        grid = GridSearchCV(xgb_classifier, parameters, n_jobs=4, scoring="neg_log_loss", cv=5, verbose=1)
        grid.fit(features, labels)

        print(f"Grid searching with K-Fold cross validation for the XGBoost is done! Best score archived with these parameters: {grid.best_params_}.")

    # train the XGBoost classifier
    elif args.type == 'xgb_train':

        print("Training the XGBoost classifier...")

        # train the XGBoost classifier
        best_parameters = {'eta': 0.05, 'gamma': 0.5, 'max_depth': 3, 'n_estimators': 100, 'subsample': 0.6}
        xgb_classifier = xgb.XGBClassifier(best_parameters)
        xgb_classifier.fit(features, labels)

        # save the classifier to .sav file
        pickle.dump(xgb_classifier, open(evaluate_folder + 'classifiers/xgboost_classifier.sav', 'wb'))

        # plot the importance table with most important features that are fed into the XGBoost
        xgb.plot_importance(xgb_classifier, max_num_features=8)
        plt.rcParams['figure.figsize'] = [2000, 1500]
        plt.savefig(cfg.path.images + 'importance_table_xgboost.svg')
        plt.show()

        # plot the visualization of first tree in the XGBoost
        graph_to_save = xgb.to_graphviz(xgb_classifier, num_trees=0, rankdir='LR')  # plot the first tree in the model
        graph_to_save.format = 'svg'
        graph_to_save.render(cfg.path.images + 'tree_0_xboost')

        # perform the K-Fold (5-Fold) cross validation on the training dataset to find out how good is our classifier
        k_fold = KFold(n_splits=5, shuffle=True)  # initialize the K-Fold (5-Fold) parameters
        results = cross_val_score(xgb_classifier, features, labels, cv=k_fold, scoring='accuracy')
        print("K-Fold Cross Validation Accuracy: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))

        print("Process of training the XGBoost classifier is done!")

    # print stats of every fold during the K-Fold cross validation
    elif args.type == 'xgb_stats':

        print("Printing the XGBoost classifier K-Fold CV training stats...")

        # prepare the XGBoost classifier and all the parameters
        best_parameters = {'eta': 0.05, 'gamma': 0.5, 'max_depth': 3, 'n_estimators': 100, 'subsample': 0.6}
        xgb_classifier = xgb.XGBClassifier(best_parameters)

        # perform the K-Fold (5-Fold) cross validation on the training dataset to find out how good is our classifier
        k_fold = KFold(n_splits=5, shuffle=True)  # initialize the K-Fold (5-Fold) parameters

        # start the process of printing stats of every fold during the K-fold cross validation
        index = 0
        for train_index, test_index in k_fold.split(features):

            # prepare train and test data for this run
            features_train, features_test = features[train_index], features[test_index]
            labels_train, labels_test = labels[train_index], labels[test_index]

            # fit the training data
            xgb_classifier.fit(features_train, labels_train)

            # evaluate and print final stats for the training dataset
            cm = confusion_matrix(labels_test, xgb_classifier.predict(features_test), labels=[0, 1, 2, 3])
            print(cm)
            print(classification_report(labels_test, xgb_classifier.predict(features_test)))

            plot_utils.plot_confusion_matrix(cm=cm, normalize=False, path=cfg.path.graphs + f'confusion_matrix_train_classifier_xgboost_{index}fold.png',
                                             title=f"Confusion matrix for the {index}. fold in the 5-fold cross validation", cmap=None,
                                             target_names=['Negative', 'ITC', 'Micro', 'Macro'])
            index += 1

        print("Process of printing the XGBoost classifier K-Fold CV training stats is done!")


def train_random_forest():
    """
    This function prepares the Random Forest classifier to predict the slide-level stage (negative/ITC/micrometastasis/
    macrometastasis) of the WSI. You can choose from three options:

        > Random Forest grid - this part performs the K-Fold (5-Fold) cross validation grid search to find optimal
        parameters for the Random Forest classifier training.
        > Random Forest train - this part trains the Random Forest classifier with optimal parameters and saves it to .sav
        file. Performance of the classifier is evaluated by the K-Fold (5-Fold) cross validation to ensure unbiased
        results.
        > Random Forest stats - this parts prints evaluation stats of every fold evaluated during the K-Fold cross
        validation. It also creates and saves confusion matrices.

    :return: None
    """

    # perform the K-Fold cross validation grid search to find best parameters of the Random Forest classifier
    if args.type == 'rf_grid':

        print("Performing the K-Fold cross validation grid search for the Random Forest...")

        # prepare the Random Forest classifier and all the parameters that should be involved in the grid searching
        rf_classifier = RandomForestClassifier()
        parameters = {
            "n_estimators": [10, 30, 50, 80, 100, 200, 300, 400, 500],  # default=100
            "max_depth": [10, 30, 50, 80, 100],  # default=None
            "min_samples_split": [0.1, 0.5, 1.0, 5, 10, 20],  # default=2
            "min_samples_leaf": [0.1, 0.5, 1, 5, 10, 15],  # default=1
            "max_features": ['auto', 'sqrt', 'log2']  # default='auto'
        }

        # perform the K-Fold (5-Fold) cross validation grid search on the training dataset
        grid = GridSearchCV(rf_classifier, parameters, n_jobs=4, scoring="neg_log_loss", cv=5, verbose=1)
        grid.fit(features, labels)

        print(f"Grid searching with K-Fold cross validation for the Random Forest is done! Best score archived with these parameters: {grid.best_params_}.")

    # train the Random Forest classifier
    elif args.type == 'rf_train':

        print("Training the Random Forest classifier...")

        # training the Random Forest classifier
        rf_classifier = RandomForestClassifier(max_depth=80, max_features='log2', min_samples_leaf=5, min_samples_split=20, n_estimators=50)
        rf_classifier.fit(features, labels)

        # save the classifier to .sav file
        pickle.dump(rf_classifier, open(evaluate_folder + 'classifiers/random_forest_classifier.sav', 'wb'))

        # perform the K-Fold (5-Fold) cross validation on the training dataset to find out how good is our classifier
        k_fold = KFold(n_splits=5, shuffle=True)  # initialize the K-Fold (5-Fold) parameters
        results = cross_val_score(rf_classifier, features, labels, cv=k_fold, scoring='accuracy')
        print("K-Fold Cross Validation Accuracy: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))

        print("Process of training the Random Forest classifier is done!")

    # print stats of every fold during the K-Fold cross validation
    elif args.type == 'rf_stats':

        print("Printing the Random Forest classifier K-Fold CV training stats...")

        # prepare the Random Forest classifier and all the parameters
        rf_classifier = RandomForestClassifier(max_depth=80, max_features='log2', min_samples_leaf=5, min_samples_split=20, n_estimators=50)

        # perform the K-Fold (5-Fold) cross validation on the training dataset to find out how good is our classifier
        k_fold = KFold(n_splits=5, shuffle=True)  # initialize the K-Fold (5-Fold) parameters

        # start the process of printing stats of every fold during the K-fold cross validation
        index = 0
        for train_index, test_index in k_fold.split(features):

            # prepare train and test data for this run
            features_train, features_test = features[train_index], features[test_index]
            labels_train, labels_test = labels[train_index], labels[test_index]

            # fit the training data
            rf_classifier.fit(features_train, labels_train)

            # evaluate and print final stats for the training dataset
            cm = confusion_matrix(labels_test, rf_classifier.predict(features_test), labels=[0, 1, 2, 3])
            print(cm)
            print(classification_report(labels_test, rf_classifier.predict(features_test)))

            plot_utils.plot_confusion_matrix(cm=cm, normalize=False, path=cfg.path.graphs + f'confusion_matrix_train_classifier_random_forest_{index}fold.png',
                                             title=f"Confusion matrix for the {index}. fold in the 5-fold cross validation", cmap=None,
                                             target_names=['Negative', 'ITC', 'Micro', 'Macro'])
            index += 1

        print("Process of printing the Random Forest classifier K-Fold CV training stats is done!")


if __name__ == "__main__":

    args = utils.parse_args()

    # prepare a dictionary with slide-level stages
    stages = {'negative': 0, 'itc': 1, 'micro': 2, 'macro': 3}

    # prepare all the necessities
    evaluate_folder = cfg.path.evaluate_patient
    slide_features_array = []
    slide_stages_array = []

    # open the C17 .csv file with reference stages of the training slides
    reference_df = pd.read_csv(cfg.path.c17_stage_labels)
    reference_map = {df_row[0]: df_row[1].lower() for _, df_row in reference_df.iterrows() if df_row[0].lower().endswith('.tif')}

    # append features and slide-level stages of every training slide to one big array
    for patient_id, reference_stage in reference_map.items():
        slide_features_array.append(np.load(evaluate_folder + 'features/features_' + patient_id[:-4] + '.npy'))
        slide_stages_array.append(stages[reference_stage])

    # convert arrays to .npy format --> prepare training dataset
    features = np.asarray(slide_features_array)
    labels = np.asarray(slide_stages_array)

    # work with the Random Forest classifier
    if 'rf' in args.type:
        train_random_forest()

    # work with the XGBoost classifier
    if 'xgb' in args.type:
        train_xgboost()
