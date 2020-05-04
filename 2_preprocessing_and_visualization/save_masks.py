"""
File:           save_masks.py
Author:         Pavlina Koutecka
Date:           02/03/2020
Description:    This file creates and saves variable WSI's mask for CAMELYON16 and CAMELYON17 dataset.
                It can distinguish between training and testing parts of dataset and also between
                normal and tumor slides. Depending on it this file creates corresponding masks and
                saves them to chosen folder.
"""

import os
from os import path
import csv

import sys
import pathlib
parent_path = str(pathlib.Path().absolute().parent)
sys.path.append(parent_path)
sys.path.append(parent_path + '/2_preprocessing_and_visualization')
import configuration as cfg
import utils
import make_masks


def save_masks_c16(slides_folder, annotations_folder, masks_folder, mask_level):
    """
    This function generates and saves all the needed masks for given folder of the CAMELYON16 dataset.
    The function can distinguish between normal and tumor slides and depending on it creates
    corresponding masks.

    :param slides_folder: path to the folder with slides which should be processed
    :param annotations_folder: path to the folder with annotations of these slides
    :param masks_folder: path to the folder where should be generated masks stored
    :param mask_level:  level to which should be the slide downsampled
    :return: None
    """

    # create list of available slides
    slides_list = [f for f in os.listdir(slides_folder) if path.isfile(path.join(slides_folder, f))]

    for slide in slides_list:

        print("Generating masks of the slide ", slide, end='\r')

        # check if the slide is not corrupted
        if slide in (cfg.wsi.c16_error_train_slides or cfg.wsi.c16_error_test_slides):
            print("Warning! Slide", slide, "is corrupted. Continuing with another slide.")
            continue

        # for every slide create mask of the slide and tissue region mask
        try:
            make_masks.make_slide_mask(slides_folder + slide, annotations_folder, masks_folder + slide.replace('.tif', '_slide_mask.png'), mask_level)
            make_masks.make_tissue_region_mask(slides_folder + slide, annotations_folder, masks_folder + slide.replace('.tif', '_tissue_region_mask.png'), mask_level)
        except:
            continue

        # if the slide contains annotated tumor region, create tumor mask, tumor boundaries mask and normal region mask too
        if annotations_folder:

            # format the name of the slide to correspond its annotation
            dic = {'T': 't', '.tif': '.xml'}
            formatted_slide = slide
            for i, j in dic.items():
                formatted_slide = formatted_slide.replace(i, j)

            try:
                make_masks.make_tumor_mask(slides_folder + slide, annotations_folder + formatted_slide, masks_folder + slide.replace('.tif', '_tumor_mask.png'), mask_level)
                make_masks.make_tumor_boundaries_mask(slides_folder + slide, annotations_folder + formatted_slide, masks_folder + slide.replace('.tif', '_tumor_boundaries.png'), mask_level)
                make_masks.make_normal_mask(slides_folder + slide, annotations_folder + formatted_slide, masks_folder + slide.replace('.tif', '_normal_mask.png'), mask_level)
            except:
                continue


def save_masks_c17(slides_folder, annotations_folder, normal_slides, masks_folder, mask_level):
    """
    This function generates and saves all the needed masks for given folder of the CAMELYON17 dataset.
    The function can distinguish between normal and tumor slides and depending on it creates
    corresponding masks.

    :param slides_folder: path to the folder with slides which should be processed
    :param annotations_folder: path to the folder with annotations of these slides
    :param normal_slides: list of slides with stage 'normal' (means no tumor training slides)
    :param masks_folder: path to the folder where should be generated masks stored
    :param mask_level:  level to which should be the slide downsampled
    :return: None
    """

    # training tumor slides
    # if the slide contains annotated tumor region, create all the masks
    if annotations_folder:

        # create list of available tumor slides
        xml_list = [f for f in os.listdir(annotations_folder) if path.isfile(path.join(annotations_folder, f))]

        # for every slide with tumor region create mask of the slide, tissue region mask, tumor mask, tumor boundaries mask and normal region mask
        for xml in xml_list:

            print("Generating masks of the slide ", xml, end='\r')

            # check if the slide is not corrupted
            if xml.replace('.xml', '.tif') in cfg.wsi.c17_error_slides:
                print("Warning! Slide", xml.replace('.xml', '.tif'), "is corrupted. Continuing with another slide.")
                continue

            try:
                make_masks.make_slide_mask(slides_folder + xml.replace('.xml', '.tif'), annotations_folder + xml, masks_folder + xml.replace('.xml', '_slide_mask.png'), mask_level)
                make_masks.make_tissue_region_mask(slides_folder + xml.replace('.xml', '.tif'), annotations_folder + xml, masks_folder + xml.replace('.xml', '_tissue_region_mask.png'), mask_level)
                make_masks.make_tumor_mask(slides_folder + xml.replace('.xml', '.tif'), annotations_folder + xml, masks_folder + xml.replace('.xml', '_tumor_mask.png'), mask_level)
                make_masks.make_tumor_boundaries_mask(slides_folder + xml.replace('.xml', '.tif'), annotations_folder + xml, masks_folder + xml.replace('.xml', '_tumor_boundaries.png'), mask_level)
                make_masks.make_normal_mask(slides_folder + xml.replace('.xml', '.tif'), annotations_folder + xml, masks_folder + xml.replace('.xml', '_normal_mask.png'), mask_level)
            except:
                continue

    # training normal slides
    # if the slide contains stage label and no tumor region, create mask of the slide and tissue region mask
    if normal_slides:

        for slide in normal_slides:

            print("Generating masks of the slide ", slide, end='\r')

            # check if the slide is not corrupted
            if slide in cfg.wsi.c17_error_slides:
                print("Warning! Slide", slide, "is corrupted. Continuing with another slide.")
                continue

            # for every slide create mask of the slide and tissue region mask
            try:
                make_masks.make_slide_mask(slides_folder + slide, annotations_folder, masks_folder + slide.replace('.tif', '_slide_mask.png'), mask_level)
                make_masks.make_tissue_region_mask(slides_folder + slide, annotations_folder, masks_folder + slide.replace('.tif', '_tissue_region_mask.png'), mask_level)
            except:
                continue

    # testing slides
    # if the slide does not contain any tumor region and stage label, create mask of the slide and tissue region mask
    else:
        # create list of available slides
        slides_list = [f for f in os.listdir(slides_folder) if path.isfile(path.join(slides_folder, f))]

        for slide in slides_list:

            print("Generating masks of the slide ", slide, end='\r')

            # check if the slide is not corrupted
            if slide in cfg.wsi.c17_error_slides:
                print("Warning! Slide", slide, "is corrupted. Continuing with another slide.")
                continue

            # for every slide create mask of the slide and tissue region mask
            try:
                make_masks.make_slide_mask(slides_folder + slide, annotations_folder, masks_folder + slide.replace('.tif', '_slide_mask.png'), mask_level)
                make_masks.make_tissue_region_mask(slides_folder + slide, annotations_folder, masks_folder + slide.replace('.tif', '_tissue_region_mask.png'), mask_level)
            except:
                continue


if __name__ == '__main__':

    args = utils.parse_args()

    # save CAMELYON16 training normal slides masks
    if args.type == 'c16_train_normal':
        save_masks_c16(cfg.path.c16_training_normal, None, cfg.path.c16_masks_training, cfg.hyperparameter.mask_level)

    # save CAMELYON16 training tumor slides masks
    if args.type == 'c16_train_tumor':
        save_masks_c16(cfg.path.c16_training_tumor, cfg.path.c16_annotations, cfg.path.c16_masks_training, cfg.hyperparameter.mask_level)

    # save CAMELYON16 testing slides masks
    if args.type == 'c16_test':
        save_masks_c16(cfg.path.c16_testing, cfg.path.c16_annotations, cfg.path.c16_masks_testing, cfg.hyperparameter.mask_level)

    # save CAMELYON17 training normal slides masks
    if args.type == 'c17_train_normal':
        with open(cfg.path.c17_stage_labels) as csvfile:
            reader = csv.reader(csvfile)
            stage_normal_slides = []
            for row in reader:
                if row[1] == 'negative':
                    stage_normal_slides.append(row[0])
        save_masks_c17(cfg.path.c17_training, None, stage_normal_slides, cfg.path.c17_masks_training, cfg.hyperparameter.mask_level)

    # save CAMELYON17 training tumor slides masks
    if args.type == 'c17_train_tumor':
        save_masks_c17(cfg.path.c17_training, cfg.path.c17_annotations, None, cfg.path.c17_masks_training, cfg.hyperparameter.mask_level)

    # save CAMELYON17 testing slides masks
    if args.type == 'c17_test':
        save_masks_c17(cfg.path.c17_testing, None, None, cfg.path.c17_masks_testing, cfg.hyperparameter.mask_level)

