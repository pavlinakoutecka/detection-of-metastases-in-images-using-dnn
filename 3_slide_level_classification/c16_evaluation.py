"""
File:           c16_evaluation.py
Author:         Pavlina Koutecka
Date:           22/04/2020
Description:    This file evaluates every CAMELYON16 test slide with regard to the official CAMELYON16 challenge
                requirements (https://camelyon16.grand-challenge.org/Evaluation/). The evalution function
                takes mask of predicted tumor region (computed by the evaluate_slide.py file), finds every
                tumor that is located on the slide (bigger than specified threshold value), computes its
                coordinates and confidence and stores it to the required .csv file. It also saves .png file with
                circled locations of tumors that are written into the .csv file.
"""


import os
from os import path
import numpy as np
import pandas as pd
import cv2
import random
from datetime import datetime
now = datetime.now()
import torch.nn.functional

import sys
import pathlib
parent_path = str(pathlib.Path().absolute().parent)
sys.path.append(parent_path)
sys.path.append(parent_path + '/2_preprocessing_and_visualization')
sys.path.append(parent_path + '/3_slide_level_classification')
import configuration as cfg
import utils
import official_evaluation

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


def find_tumor():
    """
    This function aims to find every single tumor concerning the given tumor mask. It finds every tumor located on
    the slide, computes coordinates of its center, confident score of being tumor and stores it to prepared arrays.

    :return: confidence score (probability of found tumor being tumor), x coordinates pointing to centers of
             found tumors, y coordinates pointing to centers of found tumors, mask with circled tumors
    """

    # convert the thresholded grayscale predicted tumor location map to the RGB image (to draw circles around tumors)
    predicted_tumor_mask_circled = cv2.cvtColor(predicted_tumor_mask, cv2.COLOR_GRAY2RGB)

    # prepare arrays to write into the tumor location map (afterwards used to write into the final .csv file)
    confidence = []
    coords_x, coords_y = [], []

    # find contours (tumors) in the binary thresholded image
    contours, hierarchy = cv2.findContours(predicted_tumor_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # calculate moments (center) for each contour (tumor)
    for i, contour in enumerate(contours):

        # compute size of the area occupied by the tumor
        area = cv2.contourArea(contour)

        # if the area of tumor is smaller than some defined threshold (defined by the threshold_area variable),
        # skip this tumor
        threshold_area = 40
        if area < threshold_area:
            continue

        # calculate moment (center x,y coordinate) of the tumor
        moment = cv2.moments(contour)
        if moment["m00"] != 0:  # if the center is not in the origin
            center = (int(moment["m10"] / moment["m00"]), int(moment["m01"] / moment["m00"]))
        else:  # otherwise set the coordinates as the origin
            center = (0, 0)

        # draw circles around found tumors (for better observation)
        cv2.circle(predicted_tumor_mask_circled, center, 100, (0, 0, 255), 10)

        # append tumor location to prepared arrays
        confidence.append(predicted_tumor_mask[center[1], center[0]] / 255)
        coords_x.append(center[0])
        coords_y.append(center[1])

    return confidence, coords_x, coords_y, predicted_tumor_mask_circled


if __name__ == "__main__":

    print("Creating CAMELYON16 evaluation .csv and .png files with tumor probability locations...")

    # create list of all the available test slides and corresponding .xml files
    slides_folder = cfg.path.c16_testing
    slides_list = [f for f in os.listdir(slides_folder) if path.isfile(path.join(slides_folder, f))]

    # evaluate every CAMELYON16 test slide with regard to official CAMELYON16 challenge requirements
    for index, slide_name in enumerate(slides_list):

        utils.visualize_progress(index, len(slides_list))

        # check if the slide is not corrupted
        if slide_name in cfg.wsi.c16_error_test_slides:
            print("Warning! Slide", slide_name, "is corrupted. Continuing with next slide.")
            continue

        # load the predicted tumor location mask of the slide
        predicted_tumor_mask = cv2.imread(cfg.path.evaluate + slide_name.replace('.tif', '.png'), cv2.IMREAD_GRAYSCALE)

        # find all tumors, corresponding center's x,y coordinates and confidence score on the slide
        probability, x_coordinates, y_coordinates, circled_tumor_mask = find_tumor()

        # save the data about found tumors to .csv file
        raw_data = {'Confidence': probability, 'X coordinate': x_coordinates, 'Y coordinate': y_coordinates}
        df = pd.DataFrame(raw_data)
        df.to_csv(cfg.path.evaluate + 'csv_files/' + slide_name.replace('.tif', '') + '.csv', header=True, index=False, sep=',')

        # save tumor mask with circled tumors (detected by our algorithm) to .png file
        cv2.imwrite(cfg.path.evaluate + slide_name.replace('.tif', '') + '_circled_tumor.png', circled_tumor_mask)

    official_evaluation.evaluationFROC()
