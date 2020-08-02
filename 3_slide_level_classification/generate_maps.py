"""
File:           generate_maps.py
Author:         Pavlina Koutecka
Date:           20/04/2020
Description:    This file aims to evaluate, visualize and return tumor probability maps as the result of the trained model.
                It automatically loads slides and slice it to the patches both from tumor and non-tumor region. After that,
                it evaluates model's predicted probability of every pixel being tumor and stores it into .npy file.

                After the evaluation, this file also visualizes results of the evaluation. It stores .png files with various
                visualizations:

                    > predicted tumor mask - visualized as BW mask --> black...no tumor, white...tumor
                    > tumor heatmap - visualizes probability of every pixel being tumor as RGB map
                    > visualization of TP/FP/TN/FN metric with grid - visualized as coloured masks over the original slide with grid over
                      all the patches that were processed during the evaluation part
                    > grid visualization with undetected tumor region - visualizes grid over all the patches that were processed during the
                      evaluation part; also highlights region of the undetected tumor (due to the inaccurate tissue detection algorithm).
"""


import os
from os import path
import openslide
import numpy as np
import cv2
import random
from datetime import datetime
now = datetime.now()
import albumentations
from albumentations.pytorch import ToTensor
import segmentation_models_pytorch as smp

import torch
import torchvision
import torch.nn.functional
from torch.utils.data import DataLoader, Dataset

import sys
import pathlib
parent_path = str(pathlib.Path().absolute().parent)
sys.path.append(parent_path)
sys.path.append(parent_path + '/2_preprocessing_and_visualization')
sys.path.append(parent_path + '/3_slide_level_classification')
import configuration as cfg
import utils
import train_utils
import make_masks

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


def create_patch(coordinates):
    """
    This function generates patch of given size and given top left pixel position.

    :param coordinates: coordinates of the top left pixel in the level 0 reference frame
    :return: None
    """

    patch = None

    try:
        patch = slide.read_region(coordinates, cfg.hyperparameter.patch_level, (cfg.hyperparameter.patch_size, cfg.hyperparameter.patch_size))
        #   cv2.read_region PARAMETERS:
        #       > location (tuple) – (x, y) tuple giving the top left pixel in the level 0 reference frame
        #       > level (int) – the level number
        #       > size (tuple) – (width, height) tuple giving the region size
        patch = cv2.cvtColor(np.array(patch), cv2.COLOR_RGBA2RGB)

    except Exception as e:  # if we cannot access chosen region (or some other error)
        print(e)

    return patch


def extract_patches():
    """
    This function extracts all the tissue patches of given size and level from both tumor and non-tumor region
    of chosen slide. It slides through every x and y coordinate of the slide, generates the patch and stores it and its
    coordinates to prepared arrays.
    To speed up, processed are only patches extracted with regard to the tissue region mask - patches with no tissue
    are basically skipped.

    :return: arrays with all the evaluated patches and their coordinates
    """

    # prepare arrays needed for the extraction (one to store the patches and one to store its coordinates)
    patches, coordinates = [], []

    # prepare resized tissue region mask to detect patches with tissue (to speed up - patches with no tissue are skipped)
    resized_tissue_region_mask = cv2.resize(tissue_region_mask, (patch_width, patch_height), interpolation=cv2.INTER_LINEAR)

    # slide through the x and y coordinates
    for y in range(0, patch_height, cfg.hyperparameter.patch_size):
        for x in range(0, patch_width, cfg.hyperparameter.patch_size):

            # check if we are not trying to approach non-existing location
            if x + cfg.hyperparameter.patch_size > patch_width:
                x = patch_width - cfg.hyperparameter.patch_size
            if y + cfg.hyperparameter.patch_size > patch_height:
                y = patch_height - cfg.hyperparameter.patch_size

            # rescale x and y coordinates to level 0
            top_left_corner = (x, y)
            top_left_corner_rescaled = (int(x*patch_level_factor), int(y*patch_level_factor))

            # create patch mask of tissue region starting at position top_left_corner of size patch_size
            resized_tissue_region_mask_ROI = resized_tissue_region_mask[top_left_corner[1]:top_left_corner[1] + cfg.hyperparameter.patch_size,
                                                                        top_left_corner[0]:top_left_corner[0] + cfg.hyperparameter.patch_size]

            # compute number of non-black pixels in the resized tissue region mask - if it is 0 ( = patch is completely black),
            # no tissue detected in current patch, continue with another one
            non_black_pixels = resized_tissue_region_mask_ROI.any(axis=-1).sum()
            if non_black_pixels == 0:
                continue

            # create a patch starting at position top_left_corner of size patch_size and append it and its coordinates to prepared arrays
            patch = create_patch(top_left_corner_rescaled)
            patches.append(patch)
            coordinates.append(top_left_corner)

    return patches, coordinates


def evaluate_slide():
    """
    This function evaluates model's predicted probability of every pixel being tumor (0...no tumor, 1...tumor).
    Model's prediction is evaluated from extracted patches (as in the validation phase during training).
    Final probabilities are multiplied by 255 for the purpose of visualization (0...black, 255...white).

    :return: predicted tumor mask and coordinates of evaluated patches
    """

    # prepare empty mask and array for storing the evaluated prediction
    predicted_mask = np.zeros((patch_height, patch_width))
    results = []

    # extract patches and their coordinates needed for the evaluation from the slide
    patches, coordinates = extract_patches()

    # load the patches as a dataset in required format and convert it into DataLoader format to be read properly
    eval_dataset = train_utils.EvalDataset(patches, transform)
    eval_loader = DataLoader(eval_dataset, batch_size=cfg.hyperparameter.batch_size, shuffle=False, num_workers=cfg.hyperparameter.num_workers)

    # turn the model to validation mode
    model.eval()

    # disable gradient computation
    with torch.no_grad():

        # evaluate the slide in mini-batches
        for batch_idx, patches in enumerate(eval_loader):

            # load patches from eval loader
            patches = patches.to(device)

            # apply forward pass to get model's prediction
            outputs = model(patches)['out']
            outputs = torch.nn.functional.softmax(outputs, dim=1)

            # transform the model's prediction to required format and append it to the results array
            for k in range(len(outputs)):
                predicted = 255 * outputs[k][1]
                results.append(predicted.cpu().numpy())

    # write evaluated prediction of every patch to the tumor location mask
    for k in range(0, len(coordinates)):
        predicted_mask[coordinates[k][1]:coordinates[k][1] + cfg.hyperparameter.patch_size,
                       coordinates[k][0]:coordinates[k][0] + cfg.hyperparameter.patch_size] = results[k]

    return predicted_mask, coordinates


def metric_visualization():
    """
    This function computes and visualizes chosen metric. TP/FP/TN/FN cases are computed from the segmented
    masks and after that visualized as coloured masks over the original slide.

    :return: final visualization of the metric
    """

    # compute all the possible cases that could happen during the segmentation
    TP = cv2.bitwise_and(predicted_tumor_mask_thresholded, tumor_mask)
    FP = cv2.bitwise_and(predicted_tumor_mask_thresholded, normal_mask)
    TN = cv2.bitwise_and(predicted_normal_mask, normal_mask)
    FN = cv2.bitwise_and(predicted_normal_mask, tumor_mask)

    # convert computed masks to RGB sphere
    TP = cv2.cvtColor(TP, cv2.COLOR_GRAY2RGB)
    FP = cv2.cvtColor(FP, cv2.COLOR_GRAY2RGB)
    TN = cv2.cvtColor(TN, cv2.COLOR_GRAY2RGB)
    FN = cv2.cvtColor(FN, cv2.COLOR_GRAY2RGB)

    # prepare TP colour mask
    red_mask = np.zeros((mask_height, mask_width, 3), TP.dtype)
    red_mask[:, :] = (0, 0, 255)
    TP_highlighted = cv2.bitwise_and(red_mask, TP)

    # prepare FP colour mask
    blue_mask = np.zeros((mask_height, mask_width, 3), FP.dtype)
    blue_mask[:, :] = (255, 0, 0)
    FP_highlighted = cv2.bitwise_and(blue_mask, FP)

    # prepare TN colour mask
    green_mask = np.zeros((mask_height, mask_width, 3), TN.dtype)
    green_mask[:, :] = (0, 255, 0)
    TN_highlighted = cv2.bitwise_and(green_mask, TN)

    # prepare FN colour mask
    cyan_mask = np.zeros((mask_height, mask_width, 3), FN.dtype)
    cyan_mask[:, :] = (255, 255, 0)
    FN_highlighted = cv2.bitwise_and(cyan_mask, FN)

    # draw prepared colour masks over the original slide with 0.5 transparency
    final_metric_visualization = cv2.addWeighted(src1=slide_mask, alpha=1, src2=TP_highlighted, beta=0.5, gamma=0)
    final_metric_visualization = cv2.addWeighted(src1=final_metric_visualization, alpha=1, src2=FP_highlighted, beta=0.5, gamma=0)
    final_metric_visualization = cv2.addWeighted(src1=final_metric_visualization, alpha=1, src2=TN_highlighted, beta=0.5, gamma=0)
    final_metric_visualization = cv2.addWeighted(src1=final_metric_visualization, alpha=1, src2=FN_highlighted, beta=0.5, gamma=0)

    # create grid visualization (draws squares around patches that are processed during the evaluation)
    final_metric_visualization = grid_visualization(final_metric_visualization)

    return final_metric_visualization


def undetected_tumor_visualization():
    """
    This function visualizes undetected tumor region - tissue, that is classified as tumor by the pathologist, but
    our tissue detection algorithm did not detect it as the tissue.

    :return: final visualization of the undetected tumor
    """

    # prepare mask of the undetected tumor
    undetected_tumor = tumor_mask - tissue_region_mask

    # find contours (tumors) in the binary thresholded image
    _, threshold = cv2.threshold(undetected_tumor, 220, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # convert the mask to RGB sphere
    undetected_tumor = cv2.cvtColor(undetected_tumor, cv2.COLOR_GRAY2RGB)

    # prepare colour mask
    yellow_mask = np.zeros((mask_height, mask_width, 3), undetected_tumor.dtype)
    yellow_mask[:, :] = (0, 255, 255)
    undetected_tumor_highlighted = cv2.bitwise_and(yellow_mask, undetected_tumor)

    # draw prepared colour masks with borders over the original slide with 0.5 transparency
    final_undetected_tumor_visualization = cv2.addWeighted(src1=slide_mask, alpha=1, src2=undetected_tumor_highlighted, beta=0.5, gamma=0)
    cv2.drawContours(final_undetected_tumor_visualization, contours, -1, (0, 255, 255), 2)

    # create grid visualization (draws squares around patches that are processed during the evaluation)
    final_undetected_tumor_visualization = grid_visualization(final_undetected_tumor_visualization)

    return final_undetected_tumor_visualization


def grid_visualization(slide_to_grid):
    """
    This function visualizes grid over all the patches that were processed during the evaluation part.

    :param slide_to_grid: slide to make grid on
    :return: final visualization of the grid
    """

    # prepare slide to make grid on
    final_grid_visualization = slide_to_grid

    # compute patch size rescaled to the level mask_level
    patch_size = int((cfg.hyperparameter.patch_size * patch_level_factor) / mask_level_factor)

    # draw the grid over all patches
    for i in range(len(patches_coordinates)):

        # rescale x,y coordinates to level mask_level
        x, y = int(patches_coordinates[i][0] * patch_level_factor / mask_level_factor), int(patches_coordinates[i][1] * patch_level_factor / mask_level_factor)

        # draw square around the patch
        cv2.rectangle(final_grid_visualization, (x, y), (x+patch_size, y+patch_size), (0, 0, 0), 1)

    return final_grid_visualization


if __name__ == "__main__":

    print("Generating tumor-likelihood maps and creating .npy file and visualizations...")

    # print evaluation info
    print(f'\n--- Date: {cfg.hyperparameter.date}\n'
          f'--- Model: {cfg.hyperparameter.model}\n'
          f'--- Pretrained: {cfg.hyperparameter.pretrained}\n'
          f'--- Loaded weights: {cfg.hyperparameter.trained_weights}\n'
          f'--- Patch level: {cfg.hyperparameter.patch_level}\n'
          f'--- Mask level: {cfg.hyperparameter.mask_level}\n'
          f'--- Patch size: {cfg.hyperparameter.patch_size}\n'
          f'--- Learning rate: {cfg.hyperparameter.learning_rate}\n'
          f'--- Threshold area: {cfg.hyperparameter.threshold_area}\n')

    args = utils.parse_args()

    # load data transforms
    transform = albumentations.Compose([
        ToTensor(),
    ])

    # load the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # '.to(device)' can be written also as '.cuda()'

    # load the model
    if cfg.hyperparameter.model == 'deeplabv3_resnet101':
        model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=cfg.hyperparameter.pretrained, progress=True, num_classes=cfg.hyperparameter.num_classes)
    if cfg.hyperparameter.model == 'fcn_resnet50':
        model = torchvision.models.segmentation.fcn_resnet50(pretrained=cfg.hyperparameter.pretrained, progress=True, num_classes=cfg.hyperparameter.num_classes)
    if cfg.hyperparameter.model == 'unet':
        model = smp.Unet('resnet34', classes=2)
    model.to(device)

    # load trained weights
    if cfg.hyperparameter.trained_weights:
        model.load_state_dict(torch.load(cfg.hyperparameter.trained_weights, map_location=device))

    # turn the model to validation mode
    model.eval()

    # create list of all the available slides and corresponding .xml files and evaluate folders
    if args.data == 'c16_test':  # CAMELYON16 challenge --> evaluate test slides (annotations are available)
        slides_folder = cfg.path.c16_testing
        annotations_folder = cfg.path.c16_annotations
        evaluate_folder = cfg.path.evaluate_slide
    if args.data == 'c17_train':  # CAMELYON17 challenge --> evaluate train slides (annotations are available)
        slides_folder = cfg.path.c17_training
        annotations_folder = cfg.path.c17_annotations
        evaluate_folder = cfg.path.evaluate_patient
    if args.data == 'c17_test':  # CAMELYON17 challenge --> evaluate test slides (annotations are not available)
        slides_folder = cfg.path.c17_testing
        annotations_folder = None
        evaluate_folder = cfg.path.evaluate_patient
    slides_list = [f for f in os.listdir(slides_folder) if path.isfile(path.join(slides_folder, f))]

    # if we are evaluating CAMELYON16 test data or CAMELYON17 train data
    if args.data == 'c16_test' or args.data == 'c17_train':

        # create tumor probability map for every slide
        for index, slide_name in enumerate(slides_list):

            utils.visualize_progress(index, len(slides_list))

            # evaluate only required slides
            if index < int(args.start):
                continue
            if index > int(args.end):
                break

            # check if the slide is not corrupted
            if slide_name in cfg.wsi.c16_error_slides or slide_name in cfg.wsi.c17_error_slides:
                print("Warning! Slide", slide_name, "is corrupted. Continuing with next slide.")
                continue

            # prepare all the slide necessities
            slide = openslide.OpenSlide(slides_folder + slide_name)
            patch_level_factor = float(slide.level_downsamples[cfg.hyperparameter.patch_level])
            mask_level_factor = float(slide.level_downsamples[cfg.hyperparameter.mask_level])
            patch_width, patch_height = slide.level_dimensions[cfg.hyperparameter.patch_level]
            mask_width, mask_height = slide.level_dimensions[cfg.hyperparameter.mask_level]

            # prepare downsampled version of the slide image
            slide_mask = make_masks.make_slide_mask(slides_folder + slide_name, None, None, cfg.hyperparameter.mask_level)

            # prepare tissue region mask
            tissue_region_mask = make_masks.make_tissue_region_mask(slides_folder + slide_name, None, None, cfg.hyperparameter.mask_level)

            # format the name of the slide to correspond its annotation
            dic = {'T': 't', '.tif': '.xml'}
            formatted_slide_name = slide_name
            for i, j in dic.items():
                formatted_slide_name = formatted_slide_name.replace(i, j)

            # prepare tumor mask and normal (non-tumor) mask
            try:  # if the slide contains annotated tumor region, create tumor mask and normal (non-tumor) mask
                tumor_mask = make_masks.make_tumor_mask(slides_folder + slide_name, annotations_folder + formatted_slide_name, None, cfg.hyperparameter.mask_level)
                normal_mask = make_masks.make_normal_mask(slides_folder + slide_name, annotations_folder + formatted_slide_name, None, cfg.hyperparameter.mask_level)
            except:  # otherwise create empty tumor mask; normal mask generate in the same way as the tissue region mask
                tumor_mask = np.zeros((mask_height, mask_width))  # empty tumor mask - no tumors are detected
                normal_mask = make_masks.make_tissue_region_mask(slides_folder + slide_name, None, None, cfg.hyperparameter.mask_level)  # same mask as in the tissue region mask

            # evaluate predicted tumor mask of the slide
            predicted_tumor_mask, patches_coordinates = evaluate_slide()

            # resize predicted tumor mask to smaller size - to speed up, used only for examination
            predicted_tumor_mask_resized = cv2.resize(predicted_tumor_mask, (mask_width, mask_height), interpolation=cv2.INTER_LINEAR)

            # save resized tumor mask to .npy file (for CAMELYON16 evaluation purpose)
            np.save(evaluate_folder + slide_name.replace('.tif', '.npy'), predicted_tumor_mask_resized)

            # threshold resized predicted tumor location map to get only BW mask
            _, predicted_tumor_mask_thresholded = cv2.threshold(predicted_tumor_mask_resized, 220, 255, cv2.THRESH_BINARY)

            # prepare mask of predicted non-tumor region (not only tissue!) as predicted tumor region substracted from non-tumor region mask
            # (represented by complete white mask - at the start, we are assuming that every pixel is non-tumor)
            predicted_normal_mask = np.ones((mask_height, mask_width))*255 - predicted_tumor_mask_thresholded

            # convert the grayscale masks to images of type uint8 (to provide unified array type)
            predicted_tumor_mask_resized = cv2.convertScaleAbs(predicted_tumor_mask_resized)
            predicted_tumor_mask_thresholded = cv2.convertScaleAbs(predicted_tumor_mask_thresholded)
            predicted_normal_mask = cv2.convertScaleAbs(predicted_normal_mask)
            tumor_mask = cv2.convertScaleAbs(tumor_mask)
            normal_mask = cv2.convertScaleAbs(normal_mask)

            # create the TP, FP, TN, FN metric visualization
            metric_visualization_mask = metric_visualization()

            # create undetected tumor mask (= tissue, that is classified as tumor by the pathologist, but our tissue detection algorithm did not detect it as the tissue)
            undetected_tumor_mask = undetected_tumor_visualization()

            # convert the grayscale predicted tumor location map to RGB heatmap image
            heatmap = cv2.cvtColor(predicted_tumor_mask_resized, cv2.COLOR_GRAY2RGB)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_PARULA)

            # save all the visualizations to .png files
            cv2.imwrite(evaluate_folder + slide_name.replace('.tif', '') + '.png', predicted_tumor_mask_resized)
            cv2.imwrite(evaluate_folder + slide_name.replace('.tif', '') + '_metric_visualization.png', metric_visualization_mask)
            cv2.imwrite(evaluate_folder + slide_name.replace('.tif', '') + '_undetected_tumor_grid.png', undetected_tumor_mask)
            cv2.imwrite(evaluate_folder + slide_name.replace('.tif', '') + '_heatmap.png', heatmap)

    if args.data == 'c17_test':

        # create tumor probability map for every slide
        for index, slide_name in enumerate(slides_list):

            utils.visualize_progress(index, len(slides_list))

            # evaluate only required slides
            if index < int(args.start):
                continue
            if index > int(args.end):
                break

            # check if the slide is not corrupted
            if slide_name in cfg.wsi.c17_error_slides:
                print("Warning! Slide", slide_name, "is corrupted. Continuing with next slide.")
                continue

            # prepare all the slide necessities
            slide = openslide.OpenSlide(slides_folder + slide_name)
            patch_level_factor = float(slide.level_downsamples[cfg.hyperparameter.patch_level])
            mask_level_factor = float(slide.level_downsamples[cfg.hyperparameter.mask_level])
            patch_width, patch_height = slide.level_dimensions[cfg.hyperparameter.patch_level]
            mask_width, mask_height = slide.level_dimensions[cfg.hyperparameter.mask_level]

            # prepare tissue region mask
            tissue_region_mask = make_masks.make_tissue_region_mask(slides_folder + slide_name, None, None, cfg.hyperparameter.mask_level)

            # evaluate predicted tumor mask of the slide
            predicted_tumor_mask, patches_coordinates = evaluate_slide()

            # resize predicted tumor mask to smaller size - to speed up, used only for examination
            predicted_tumor_mask_resized = cv2.resize(predicted_tumor_mask, (mask_width, mask_height), interpolation=cv2.INTER_LINEAR)

            # save resized tumor mask to .npy and .png file
            np.save(evaluate_folder + slide_name.replace('.tif', '.npy'), predicted_tumor_mask_resized)
            cv2.imwrite(evaluate_folder + slide_name.replace('.tif', '') + '.png', predicted_tumor_mask_resized)
