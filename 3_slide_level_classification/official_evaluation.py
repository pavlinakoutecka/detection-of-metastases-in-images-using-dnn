# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 14:09:32 2016

@author: Babak Ehteshami Bejnordi

Evaluation code for the Camelyon16 challenge on cancer metastases detecion
"""

import openslide
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as nd
from skimage import measure
import pandas as pd
import os
from os import path

import sys
import pathlib
parent_path = str(pathlib.Path().absolute().parent)
sys.path.append(parent_path)
sys.path.append(parent_path + '/2_preprocessing_and_visualization')
sys.path.append(parent_path + '/3_slide_level_classification')
import configuration as cfg
import utils
import make_masks


def computeEvaluationMask(slideDIR, xmlDIR, resolution, level):
    """Computes the evaluation mask.

    Args:
        maskDIR:    the directory of the ground truth mask
        resolution: Pixel resolution of the image at level 0
        level:      The level at which the evaluation mask is made

    Returns:
        evaluation_mask
    """

    slide = openslide.open_slide(slideDIR)
    dims = slide.level_dimensions[level]
    pixelarray = np.zeros(dims[0]*dims[1], dtype='uint')

    pixelarray = make_masks.make_tumor_mask(slideDIR, xmlDIR, None, level)
    distance = nd.distance_transform_edt(255 - pixelarray[:,:])

    # pixelarray = np.array(slide.read_region((0,0), level, dims))
    # distance = nd.distance_transform_edt(255 - pixelarray[:,:,0])

    Threshold = 75/(resolution * pow(2, level) * 2)  # 75µm is the equivalent size of 5 tumor cells
    binary = distance < Threshold
    filled_image = nd.morphology.binary_fill_holes(binary)
    evaluation_mask = measure.label(filled_image, connectivity = 2)

    return evaluation_mask


def computeITCList(evaluation_mask, resolution, level):
    """Compute the list of labels containing Isolated Tumor Cells (ITC)

    Description:
        A region is considered ITC if its longest diameter is below 200µm.
        As we expanded the annotations by 75µm, the major axis of the object
        should be less than 275µm to be considered as ITC (Each pixel is
        0.243µm*0.243µm in level 0). Therefore the major axis of the object
        in level 5 should be less than 275/(2^5*0.243) = 35.36 pixels.

    Args:
        evaluation_mask:    The evaluation mask
        resolution:         Pixel resolution of the image at level 0
        level:              The level at which the evaluation mask was made

    Returns:
        Isolated_Tumor_Cells: list of labels containing Isolated Tumor Cells
    """
    max_label = np.amax(evaluation_mask)
    properties = measure.regionprops(evaluation_mask)
    Isolated_Tumor_Cells = []
    threshold = 275/(resolution * pow(2, level))
    for i in range(0, max_label):
        if properties[i].major_axis_length < threshold:
            Isolated_Tumor_Cells.append(i+1)
    return Isolated_Tumor_Cells


def readCSVContent(csvDIR):
    """Reads the data inside CSV file

    Args:
        csvDIR:    The directory including all the .csv files containing the results.
        Note that the CSV files should have the same name as the original image

    Returns:
        Probs:      list of the Probabilities of the detected lesions
        Xcorr:      list of X-coordinates of the lesions
        Ycorr:      list of Y-coordinates of the lesions
    """

    Xcorr, Ycorr, Probs = ([] for i in range(3))
    csv_lines = open(csvDIR, "r").readlines()

    for i in range(1, len(csv_lines)):

        line = csv_lines[i]
        elems = line.rstrip().split(',')

        x_corr = int(elems[1])
        y_corr = int(elems[2])

        Probs.append(float(elems[0]))
        Xcorr.append(x_corr)
        Ycorr.append(y_corr)

    return Probs, Xcorr, Ycorr


def compute_FP_TP_Probs(Ycorr, Xcorr, Probs, is_tumor, evaluation_mask, Isolated_Tumor_Cells, level):
    """Generates true positive and false positive stats for the analyzed image

    Args:
        Probs:      list of the Probabilities of the detected lesions
        Xcorr:      list of X-coordinates of the lesions
        Ycorr:      list of Y-coordinates of the lesions
        is_tumor:   A boolean variable which is one when the case cotains tumor
        evaluation_mask:    The evaluation mask
        Isolated_Tumor_Cells: list of labels containing Isolated Tumor Cells
        level:      The level at which the evaluation mask was made

    Returns:
        FP_probs:   A list containing the probabilities of the false positive detections

        TP_probs:   A list containing the probabilities of the True positive detections

        NumberOfTumors: Number of Tumors in the image (excluding Isolate Tumor Cells)

        detection_summary:   A python dictionary object with keys that are the labels
        of the lesions that should be detected (non-ITC tumors) and values
        that contain detection details [confidence score, X-coordinate, Y-coordinate].
        Lesions that are missed by the algorithm have an empty value.

        FP_summary:   A python dictionary object with keys that represent the
        false positive finding number and values that contain detection
        details [confidence score, X-coordinate, Y-coordinate].
    """

    max_label = np.amax(evaluation_mask)
    FP_probs = []
    TP_probs = np.zeros((max_label,), dtype=np.float32)
    detection_summary = {}
    FP_summary = {}
    for i in range(1,max_label+1):
        if i not in Isolated_Tumor_Cells:
            label = 'Label ' + str(i)
            detection_summary[label] = []

    FP_counter = 0
    if (is_tumor):
        for i in range(0, len(Xcorr)):

            HittedLabel = evaluation_mask[int(Ycorr[i]), int(Xcorr[i])]

            if HittedLabel == 0:
                FP_probs.append(Probs[i])
                key = 'FP ' + str(FP_counter)
                FP_summary[key] = [Probs[i], Xcorr[i], Ycorr[i]]
                FP_counter += 1
            elif HittedLabel not in Isolated_Tumor_Cells:
                if (Probs[i]>TP_probs[HittedLabel-1]):
                    label = 'Label ' + str(HittedLabel)
                    detection_summary[label] = [Probs[i], Xcorr[i], Ycorr[i]]
                    TP_probs[HittedLabel-1] = Probs[i]

    else:
        for i in range(0,len(Xcorr)):
            FP_probs.append(Probs[i])
            key = 'FP ' + str(FP_counter)
            FP_summary[key] = [Probs[i], Xcorr[i], Ycorr[i]]
            FP_counter+=1

    num_of_tumors = max_label - len(Isolated_Tumor_Cells);

    return FP_probs, TP_probs, num_of_tumors, detection_summary, FP_summary


def computeFROC(FROC_data):
    """Generates the data required for plotting the FROC curve

    Args:
        FROC_data:      Contains the list of TPs, FPs, number of tumors in each image

    Returns:
        total_FPs:      A list containing the average number of false positives
        per image for different thresholds

        total_sensitivity:  A list containig overall sensitivity of the system
        for different thresholds
    """

    unlisted_FPs = [item for sublist in FROC_data[1] for item in sublist]
    unlisted_TPs = [item for sublist in FROC_data[2] for item in sublist]

    total_FPs, total_TPs = [], []
    all_probs = sorted(set(unlisted_FPs + unlisted_TPs))
    for Thresh in all_probs[1:]:
        total_FPs.append((np.asarray(unlisted_FPs) >= Thresh).sum())
        total_TPs.append((np.asarray(unlisted_TPs) >= Thresh).sum())
    total_FPs.append(0)
    total_TPs.append(0)
    total_FPs = np.asarray(total_FPs)/float(len(FROC_data[0]))
    total_sensitivity = np.asarray(total_TPs)/float(sum(FROC_data[3]))

    return total_FPs, total_sensitivity


def plotFROC(total_FPs, total_sensitivity, FROC_score):
    """Plots the FROC curve

    Args:
        total_FPs:      A list containing the average number of false positives
        per image for different thresholds

        total_sensitivity:  A list containig overall sensitivity of the system
        for different thresholds

    Returns:
        -
    """
    fig = plt.figure()
    plt.xlabel('Average number of False Positives', fontsize=12)
    plt.ylabel('Metastasis detection sensitivity', fontsize=12)
    fig.suptitle('Free response receiver operating characteristic curve', fontsize=12)
    plt.plot(total_FPs, total_sensitivity, '-', color='#000000', label='FROC curve (Final score = {:.4f}'.format(FROC_score))
    plt.legend(loc='lower right')
    plt.grid(which='both')
    plt.savefig(cfg.path.graphs + f'Evaluation_FROC_lvl_{cfg.hyperparameter.patch_level}.png')
    plt.show()


def evaluationFROC():

    result_folder = cfg.path.evaluate_slide + 'csv_files/'
    slide_folder = cfg.path.c16_testing
    xml_folder = cfg.path.c16_annotations

    result_file_list = [each for each in os.listdir(result_folder) if each.endswith('.csv')]
    xml_file_list = [f for f in os.listdir(xml_folder) if path.isfile(path.join(xml_folder, f))]

    EVALUATION_MASK_LEVEL = cfg.hyperparameter.mask_level  # Image level at which the evaluation is done
    L0_RESOLUTION = 0.243  # pixel resolution at level 0

    FROC_data = np.zeros((4, len(result_file_list)), dtype=np.object)
    FP_summary = np.zeros((2, len(result_file_list)), dtype=np.object)
    detection_summary = np.zeros((2, len(result_file_list)), dtype=np.object)

    caseNum = 0
    for case in result_file_list:

        print('Evaluating performance on image:', case[0:-4], end='\r')
        sys.stdout.flush()
        csvDIR = os.path.join(result_folder, case)
        Probs, Xcorr, Ycorr = readCSVContent(csvDIR)

        # format the name of the slide to correspond its annotation
        dic = {'T': 't', '.csv': '.xml'}
        formatted_slide = case
        for i, j in dic.items():
            formatted_slide = formatted_slide.replace(i, j)

        is_tumor = True if formatted_slide in xml_file_list else False
        if is_tumor:
            slideDIR = os.path.join(slide_folder, case.replace('.csv', '.tif'))
            xmlDIR = os.path.join(xml_folder, formatted_slide)
            evaluation_mask = computeEvaluationMask(slideDIR, xmlDIR, L0_RESOLUTION, EVALUATION_MASK_LEVEL)
            ITC_labels = computeITCList(evaluation_mask, L0_RESOLUTION, EVALUATION_MASK_LEVEL)
        else:
            evaluation_mask = 0
            ITC_labels = []

        FROC_data[0][caseNum] = case
        FP_summary[0][caseNum] = case
        detection_summary[0][caseNum] = case
        FROC_data[1][caseNum], FROC_data[2][caseNum], FROC_data[3][caseNum], detection_summary[1][caseNum], FP_summary[1][caseNum] = compute_FP_TP_Probs(Ycorr, Xcorr, Probs, is_tumor, evaluation_mask, ITC_labels, EVALUATION_MASK_LEVEL)

        caseNum += 1

    # compute FROC curve
    total_FPs, total_sensitivity = computeFROC(FROC_data)

    # compute FROC score
    index_quarter = utils.find_nearest(total_FPs, 0.25)
    index_half = utils.find_nearest(total_FPs, 0.5)
    index_1 = utils.find_nearest(total_FPs, 1)
    index_2 = utils.find_nearest(total_FPs, 2)
    index_4 = utils.find_nearest(total_FPs, 4)
    index_8 = utils.find_nearest(total_FPs, 8)
    FROC_score = (total_sensitivity[index_quarter] + total_sensitivity[index_half] + total_sensitivity[index_1] +
                  total_sensitivity[index_2] + total_sensitivity[index_4] + total_sensitivity[index_8]) / 6

    # print FROC score
    print("\nFPs/WSI \t 1/4 \t\t 1/2 \t\t 1 \t\t 2 \t\t 4 \t\t 8")
    print("Sensitivity \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \t ".format(total_sensitivity[index_quarter], total_sensitivity[index_half],
                                                                                               total_sensitivity[index_1], total_sensitivity[index_2],
                                                                                               total_sensitivity[index_4], total_sensitivity[index_8]))
    print(f"\nFinal FROC score = {FROC_score}")

    # file = open(cfg.hyperparameter.saving_folder + cfg.hyperparameter.threshold_saving_string, "a")
    # file.write("\nFPs/WSI \t 1/4 \t\t 1/2 \t\t 1 \t\t 2 \t\t 4 \t\t 8\n")
    # file.write("Sensitivity \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \t \n".format(total_sensitivity[index_quarter], total_sensitivity[index_half],
    #                                                                                            total_sensitivity[index_1], total_sensitivity[index_2],
    #                                                                                            total_sensitivity[index_4], total_sensitivity[index_8]))
    # file.write(f"\nFinal FROC score = {FROC_score}\n\n")
    # file.close()

    # plot FROC curve
    plotFROC(total_FPs, total_sensitivity, FROC_score)

    # save FROC curve
    raw_data = {'total_FPs': total_FPs, 'X total_sensitivity': total_sensitivity, 'FROC_score': FROC_score}
    df = pd.DataFrame(raw_data)
    df.to_csv(cfg.path.saving_folder + f'FROC_data_lvl_{cfg.hyperparameter.patch_level}.csv', header=True, index=False, sep=',')
