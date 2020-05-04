"""
File:           save_patches.py
Author:         Pavlina Koutecka
Date:           04/03/2020
Description:    This file creates patches extracted from specified set of WSIs.
                These patches are extracted and stored in prepared folder and are used for creating
                special dataset file, as described in 'create_dataset.py'.

                You can create dataset with slides from CAMELYON16 or CAMELYON17 database.
                Depending on chosen slides, train or test dataset can be created.
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
import make_patches


def create_train_dataset_c16(slides_folder, annotations_folder):
    """
    This function creates collection of patches, called dataset, from specified WSIs. These
    patches are stored in prepared default folder and are used for creating special dataset
    file, as described in 'create_dataset.py'.
    WSIs used in this function should be train slides from CAMELYON16 database.

    :param slides_folder: path to the folder where are stored slides for creating the dataset
    :param annotations_folder: path to the folder where are stored annotations of the slide
    :return: None
    """

    # create list of available slides
    slides_list = [f for f in os.listdir(slides_folder) if path.isfile(path.join(slides_folder, f))]

    for slide in slides_list:

        print("Generating patches of the slide ", slide, end='\n')

        # if the slide contains annotated tumor region, create normal and tumor patches
        if annotations_folder:

            # format the name of the slide to correspond its annotation
            dic = {'T': 't', '.tif': '.xml'}
            formatted_slide = slide
            for i, j in dic.items():
                formatted_slide = formatted_slide.replace(i, j)

            # check if the tumor slide is exhaustively annotated
            if slide in cfg.wsi.c16_error_train_slides:  # if the slide is not exhaustively annotated, create only tumor patches
                print("Warning! Slide", slide, "is not exhaustively annotated, normal patches will not be created.")

                make_patches.sample_tumor_region(slides_folder + slide, annotations_folder + formatted_slide, slide, cfg.path.c16_patches_training,
                                                 cfg.path.c16_patches_masks_training,
                                                 cfg.hyperparameter.mask_level, cfg.hyperparameter.patch_level, cfg.hyperparameter.patch_size,
                                                 cfg.hyperparameter.tumor_patches_per_slide, None)

                continue

            else:  # if the slide is exhaustively annotated, create normal and tumor patches
                make_patches.sample_tumor_region(slides_folder + slide, annotations_folder + formatted_slide, slide, cfg.path.c16_patches_training,
                                                 cfg.path.c16_patches_masks_training,
                                                 cfg.hyperparameter.mask_level, cfg.hyperparameter.patch_level, cfg.hyperparameter.patch_size,
                                                 cfg.hyperparameter.tumor_patches_per_slide, None)

                make_patches.sample_normal_region(slides_folder + slide, annotations_folder + formatted_slide, slide, cfg.path.c16_patches_training,
                                                  cfg.path.c16_patches_masks_training,
                                                  cfg.hyperparameter.mask_level, cfg.hyperparameter.patch_level, cfg.hyperparameter.patch_size,
                                                  cfg.hyperparameter.normal_patches_per_slide, None)

        # otherwise, create only normal patches
        else:
            # TODO: normal_086 pote presunout a tuto podminku odstranit
            if slide in cfg.wsi.c16_error_train_slides:  # if the slide is not exhaustively annotated, don't create normal patches
                print("Warning! Slide", slide, "is not correct, normal patches will not be created.")
                continue

            make_patches.sample_normal_region(slides_folder + slide, annotations_folder, slide, cfg.path.c16_patches_training, cfg.path.c16_patches_masks_training,
                                              cfg.hyperparameter.mask_level, cfg.hyperparameter.patch_level, cfg.hyperparameter.patch_size,
                                              cfg.hyperparameter.normal_patches_per_slide, None)


def create_test_dataset_c16(slides_folder, annotations_folder):
    """
    This function creates collection of patches, called dataset, from specified WSIs. These
    patches are stored in prepared default folder and are used for creating special dataset
    file, as described in 'create_dataset.py'.
    WSIs used in this function should be test slides from CAMELYON16 database.

    :param slides_folder: path to the folder where are stored slides for creating the dataset
    :param annotations_folder: path to the folder where are stored annotations of the slide
    :return: None
    """

    # create list of all the available test slides
    slides_list = [f for f in os.listdir(slides_folder) if path.isfile(path.join(slides_folder, f))]

    # create list of available test slides with tumor annotation
    xml_list = [f for f in os.listdir(annotations_folder) if path.isfile(path.join(annotations_folder, f))]

    for slide in slides_list:

        print("Generating patches of the slide ", slide, end='\n')

        # check if the slide is not corrupted
        if slide in cfg.wsi.c16_error_test_slides:
            print("Warning! Slide", slide, "is corrupted. Continuing with next slide.")
            continue

        # format the name of the slide to correspond its annotation
        dic = {'T': 't', '.tif': '.xml'}
        formatted_slide = slide
        for i, j in dic.items():
            formatted_slide = formatted_slide.replace(i, j)

        # check if the slide contains tumor region (tumor annotation is available)
        if formatted_slide in xml_list:  # if slide contains tumor region, create both normal and tumor patches
            make_patches.sample_normal_region(slides_folder + slide, annotations_folder + formatted_slide, slide, cfg.path.c16_patches_testing,
                                              cfg.path.c16_patches_masks_testing,
                                              cfg.hyperparameter.mask_level, cfg.hyperparameter.patch_level, cfg.hyperparameter.patch_size,
                                              cfg.hyperparameter.normal_patches_per_slide, None)
            make_patches.sample_tumor_region(slides_folder + slide, annotations_folder + formatted_slide, slide, cfg.path.c16_patches_testing, cfg.path.c16_patches_masks_testing,
                                             cfg.hyperparameter.mask_level, cfg.hyperparameter.patch_level, cfg.hyperparameter.patch_size,
                                             cfg.hyperparameter.tumor_patches_per_slide, None)

        else:  # otherwise, create only normal patches
            make_patches.sample_normal_region(slides_folder + slide, None, slide, cfg.path.c16_patches_testing,
                                              cfg.path.c16_patches_masks_testing,
                                              cfg.hyperparameter.mask_level, cfg.hyperparameter.patch_level, cfg.hyperparameter.patch_size,
                                              cfg.hyperparameter.normal_patches_per_slide, None)


def create_train_dataset_c17(slides_folder, annotations_folder, normal_slides):
    """
    This function creates collection of patches, called dataset, from specified WSIs. These
    patches are stored in prepared default folder and are used for creating special dataset
    file, as described in 'create_dataset.py'.
    WSIs used in this function should be train slides from CAMELYON17 database.

    :param slides_folder: path to the folder where are stored slides for creating the dataset
    :param annotations_folder: path to the folder where are stored annotations of the slide
    :param normal_slides: list of slides with stage 'normal' (means no tumor training slides)
    :return: None
    """

    # training tumor slides
    # if the slide contains annotated tumor region, create both normal and tumor patches
    if annotations_folder:

        # create list of available tumor slides
        xml_list = [f for f in os.listdir(annotations_folder) if path.isfile(path.join(annotations_folder, f))]

        # for every slide with tumor region create mask of the slide, tissue region mask, tumor mask, tumor boundaries mask and normal region mask
        for xml in xml_list:

            print("Generating patches of the slide ", xml, end='\n')

            # check if the slide is not corrupted
            if xml.replace('.xml', '.tif') in cfg.wsi.c17_error_slides:
                print("Warning! Slide", xml.replace('.xml', '.tif'), "is corrupted. Continuing with next slide.")
                continue

            # create normal and tumor patches
            make_patches.sample_normal_region(slides_folder + xml.replace('.xml', '.tif'), annotations_folder + xml, xml.replace('.xml', '.tif'), cfg.path.c17_patches_training,
                                              cfg.path.c17_patches_masks_training,
                                              cfg.hyperparameter.mask_level, cfg.hyperparameter.patch_level, cfg.hyperparameter.patch_size,
                                              cfg.hyperparameter.normal_patches_per_slide, None)
            make_patches.sample_tumor_region(slides_folder + xml.replace('.xml', '.tif'), annotations_folder + xml, xml.replace('.xml', '.tif'),
                                             cfg.path.c17_patches_training, cfg.path.c17_patches_masks_training,
                                             cfg.hyperparameter.mask_level, cfg.hyperparameter.patch_level, cfg.hyperparameter.patch_size,
                                             cfg.hyperparameter.tumor_patches_per_slide, None)

    # training normal slides
    # if the slide contains stage label and no tumor region, create only normal patches
    if normal_slides:

        for slide in normal_slides:

            print("Generating patches of the slide ", slide, end='\n')

            # check if the slide is not corrupted
            if slide in cfg.wsi.c17_error_slides:
                print("Warning! Slide", slide, "is corrupted. Continuing with another slide.")
                continue

            # create normal patches
            make_patches.sample_normal_region(slides_folder + slide, annotations_folder, slide, cfg.path.c17_patches_training,
                                              cfg.path.c17_patches_masks_training,
                                              cfg.hyperparameter.mask_level, cfg.hyperparameter.patch_level, cfg.hyperparameter.patch_size,
                                              cfg.hyperparameter.normal_patches_per_slide, None)


if __name__ == '__main__':

    args = utils.parse_args()

    # create CAMELYON16 training normal slides patches
    if args.type == 'c16_train_normal':
        create_train_dataset_c16(cfg.path.c16_training_normal, None)

    # create CAMELYON16 training tumor slides patches
    if args.type == 'c16_train_tumor':
        create_train_dataset_c16(cfg.path.c16_training_tumor, cfg.path.c16_annotations)

    # create CAMELYON16 testing slides patches
    if args.type == 'c16_test':
        create_test_dataset_c16(cfg.path.c16_testing, cfg.path.c16_annotations)

    # create CAMELYON17 training normal slides patches
    if args.type == 'c17_train_normal':
        with open(cfg.path.c17_stage_labels) as csvfile:
            reader = csv.reader(csvfile)
            stage_normal_slides = []
            for row in reader:
                if row[1] == 'negative':
                    stage_normal_slides.append(row[0])
        create_train_dataset_c17(cfg.path.c17_training, None, stage_normal_slides)

    # create CAMELYON17 training tumor slides patches
    if args.type == 'c17_train_tumor':
        create_train_dataset_c17(cfg.path.c17_training, cfg.path.c17_annotations, None)
