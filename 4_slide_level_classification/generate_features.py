"""
File:           generate_features.py
Author:         Pavlina Koutecka
Date:           25/05/2020
Description:    This file generates chosen features from the tumor probability maps and stores them to the .npy file.
                At the beginning, the tumor probability map is thresholded to BW mask at three different threshold
                levels - 50%, 90% and 99%. After that, chosen features of the mask are extracted for every thresholding level
                and stored to .npy file. Special .npy file with features extracted only at the 99% level is also created.
"""


import os
from os import path
import numpy as np
import cv2
import skimage
import random
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


def extract_features():
    """
    This function generates chosen features from the tumor probability maps and stores them to the .npy file.
    At the beginning, the tumor probability map is thresholded to BW mask at three different threshold
    levels - 50%, 90% and 99%. After that, chosen features of the mask are extracted for every thresholding level
    and stored to .npy file. Special .npy file with features extracted only at the 99% level is also created.

    FEATURES THAT ARE EXTRACTED:
        1. Area of largest connected region                      + mean
        2. Major axis length of largest connected region         + max, mean
        3. Perimeter of largest connected region                 + max, mean
        4. Max value of largest connected region                 + max, mean
        5. Mean value of largest connected region                + max, mean
        6. Eccentricity of largest connected region              + max, mean
        7. Extent of largest connected region                    + max, mean
        8. Solidity of largest connected region                  + max, mean
        9. Number of connected regions
        10. Area predicted as tumor in total

    :return: None
    """

    # prepare thresholds for thresholding at three different levels - 50%, 90% and 99%
    thresholds = [128, 242, 254]

    # prepare array for all the final extracted features (for all three thresholds) of one slide
    final_features = []

    # load the predicted tumor location mask of the slide
    predicted_tumor_mask = np.load(evaluate_folder + slide_name.replace('.tif', '.npy'))

    # extract features for every threshold value
    for i, threshold in enumerate(thresholds):

        # prepare array for extracted features
        features = np.empty(25)

        # make thresholded tumor mask and convert it to image of type uint8
        _, predicted_tumor_mask_thresholded = cv2.threshold(predicted_tumor_mask, threshold, 255, cv2.THRESH_BINARY)
        predicted_tumor_mask_thresholded = cv2.convertScaleAbs(predicted_tumor_mask_thresholded)

        # extract all the connected components and take their properties
        labels = skimage.measure.label(predicted_tumor_mask_thresholded, background=0)
        properties = skimage.measure.regionprops(labels, intensity_image=predicted_tumor_mask_thresholded)

        # if there are some properties, compute features
        if len(properties) > 0:

            # find area of the largest connected region (tumor) and index of this region in the properties array
            max_area, max_index = max([(p.area, idx) for idx, p in enumerate(properties)])

            # compute features
            features[0], features[1] = \
                properties[max_index].area, sum(p.area for p in properties)/len(properties)
            features[2], features[3], features[4] = \
                properties[max_index].major_axis_length, max(p.major_axis_length for p in properties), sum(p.major_axis_length for p in properties)/len(properties)
            features[5], features[6], features[7] = \
                properties[max_index].perimeter, max(p.perimeter for p in properties), sum(p.perimeter for p in properties)/len(properties)
            features[8], features[9], features[10] = \
                properties[max_index].max_intensity, max(p.max_intensity for p in properties), sum(p.max_intensity for p in properties)/len(properties)
            features[11], features[12], features[13] = \
                properties[max_index].mean_intensity, max(p.mean_intensity for p in properties), sum(p.mean_intensity for p in properties)/len(properties)
            features[14], features[15], features[16] = \
                properties[max_index].eccentricity, max(p.eccentricity for p in properties), sum(p.eccentricity for p in properties)/len(properties)
            features[17], features[18], features[19] = \
                properties[max_index].extent, max(p.extent for p in properties), sum(p.extent for p in properties)/len(properties)
            features[20], features[21], features[22] = \
                properties[max_index].solidity, max(p.solidity for p in properties), sum(p.solidity for p in properties)/len(properties)
            features[23] = len(properties)
            features[24] = sum(p.area for p in properties)

        # append features extracted for this threshold to the final features array
        final_features = np.append(final_features, features)

    # save two .npy files with different features - one with features extracted at all three thresholding levels and one with features extracted only at the 99% thresholding level
    np.save(evaluate_folder + 'features/features_' + slide_name.replace('.tif', '.npy'), final_features)  # extracted at 50%, 90% and 99% thresholding level
    np.save(evaluate_folder + 'features/features_simple_' + slide_name.replace('.tif', '.npy'), features)  # extracted only at 99% thresholding level


if __name__ == "__main__":

    args = utils.parse_args()

    print("Extracting features from the generated tumor-likelihood maps...")

    # create list of needed folders and all the available train/test slides
    if args.data == 'c17_train':
        slides_folder = cfg.path.c17_training
    if args.data == 'c17_test':
        slides_folder = cfg.path.c17_testing
    evaluate_folder = cfg.path.evaluate_patient
    slides_list = [f for f in os.listdir(slides_folder) if path.isfile(path.join(slides_folder, f))]

    # extract features from every train/test slide
    for index, slide_name in enumerate(slides_list):

        utils.visualize_progress(index, len(slides_list))

        # check if the slide is not corrupted
        if slide_name in cfg.wsi.c17_error_slides:
            print("Warning! Slide", slide_name, "is corrupted. Continuing with next slide.")
            continue

        extract_features()

    print("Process of extracting features from the generated tumor-likelihood maps is done!")
