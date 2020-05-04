# -*- coding: utf-8 -*-
"""
File:           create_dataset.py
Author:         Pavlina Koutecka
Date:           05/03/2020
Description:    This file converts created patches into one big dataset. Currently are implemented
                two different ways to store the dataset:
                    > h5py format - converts prepared patches into one .h5py file with all the patches
                    and and one .h5py file with their masks; this way of storage should speed up the
                    process of loading the dataset during training; currently not in usage
                    > csv format - writes paths to prepared patches into one .csv file with all the
                    paths to patches and their masks; easy to use, fast even with a lot of data;
                    currently in usage

                While using csv format, you can create training (used for personal training), validation
                (used for personal validation) or final (used for the final training) dataset's .csv
                files.

                Folders, that are included in creating training dataset:
                    > CAMELYON16 training patches (cfg.path.c16_patches_training, cfg.path.c16_patches_masks_training)
                    > CAMELYON16 testing patches (cfg.path.c16_patches_testing, cfg.path.c16_patches_masks_testing)
                Folders, that are included in creating validation dataset:
                    > CAMELYON16 testing patches (cfg.path.c16_patches_testing, cfg.path.c16_patches_masks_testing)
                Folders, that are included in creating final dataset:
                    > CAMELYON16 training patches (cfg.path.c16_patches_training, cfg.path.c16_patches_masks_training)
                    > CAMELYON16 testing patches (cfg.path.c16_patches_testing, cfg.path.c16_patches_masks_testing)
                    > CAMELYON17 training patches (cfg.path.c17_patches_training, cfg.path.c17_patches_masks_training)
"""

import os
from os import path
import cv2
import numpy as np
import h5py
import pandas as pd

import sys
import pathlib
parent_path = str(pathlib.Path().absolute().parent)
sys.path.append(parent_path)
import configuration as cfg
import utils


def convert_to_h5py(path_dataset_patch, path_dataset_mask, folder_patch, folder_mask, create_file=False):
    """
    This function converts created patches into dataset in .h5py format. It creates one file for patches
    and one for their masks.
    Patches and masks are due to the memory restrictions written into the dataset file in bunches, not
    all at one time.

    :param path_dataset_patch: path to where should be stored created dataset with patches
    :param path_dataset_mask: path to where should be stored created dataset with patches masks
    :param folder_patch: path to the folder where are stored created patches
    :param folder_mask: path to the folder where are stored patches masks
    :param create_file: 'True' if should be created completely new dataset file, 'False' if we
                         load already existing dataset file
    :return: None
    """

    # create list of available patches and their masks in the right order
    patches_list = [f for f in os.listdir(folder_patch) if path.isfile(path.join(folder_patch, f))]
    masks_list = [patch.replace('patch', 'patch_mask') for patch in patches_list]

    # specify how big should be bunches for writing into the dataset file
    length = int(len(patches_list))
    bunch = int(length / 10)
    print("Total size of patches in dataset after conversion will be", length)

    # write all patches and masks to the dataset bunch after bunch
    for i in range(0, length, bunch):

        # check if we are not at the end of the patches list
        end_i = i + bunch
        if end_i > length:
            end_i = length

        print("\nConverting patches and masks from number", i, "to number", end_i)

        # prepare part of the patches and masks to be written into the dataset file
        patches_list_part = patches_list[i:end_i]
        masks_list_part = masks_list[i:end_i]

        # prepare arrays for storing patches and their masks
        patches_bunch = []
        masks_bunch = []

        # write patches and masks in one bunch to the dataset
        for index, patch in enumerate(patches_list_part):

            utils.visualize_progress(index, len(patches_list_part))

            # load the patch
            try:
                img = cv2.imread(folder_patch + patches_list_part[index])
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            except:
                continue

            # load corresponding mask
            try:
                img_mask = cv2.imread(folder_mask + masks_list_part[index])
                img_mask = cv2.cvtColor(img_mask, cv2.COLOR_BGR2RGB)
            except:
                continue

            # append patch and its mask to patches and masks array
            patches_bunch.append(img)
            masks_bunch.append(img_mask)

        # convert patches and masks array into numpy array
        patches = np.array(patches_bunch, dtype='uint8')
        masks = np.array(masks_bunch, dtype='uint8')

        # create completely new dataset file and write into it bunch of patches and masks
        if i == 0 and create_file:
            # save patches dataset as one big file in h5py format
            # (no need to close the file because of the with statement)
            with h5py.File(path_dataset_patch, 'w') as hf:
                hf.create_dataset('1', data=patches, maxshape=(None, None, None, None))
                #   hf.create_dataset PARAMETERS:
                #       > name - specifies dataset's name
                #       > data - data that should be stored in this dataset
                #       > maxshape - maximum size of this dataset

            # save masks dataset as one big file in h5py format
            # (no need to close the file because of the with statement)
            with h5py.File(path_dataset_mask, 'w') as hf:
                hf.create_dataset('1', data=masks, maxshape=(None, None, None, None))

        # load already existing dataset file and write into it bunch of patches and masks
        else:
            with h5py.File(path_dataset_patch, 'a') as hf:
                hf['1'].resize((hf['1'].shape[0] + patches.shape[0]), axis=0)
                hf['1'][-patches.shape[0]:] = patches

            with h5py.File(path_dataset_mask, 'a') as hf:
                hf['1'].resize((hf['1'].shape[0] + masks.shape[0]), axis=0)
                hf['1'][-masks.shape[0]:] = masks


if __name__ == '__main__':

    args = utils.parse_args()

    # add CAMELYON16 training dataset, CAMELYON16 testing dataset and CAMELYON17 training dataset to the general training dataset in .h5py format
    if args.type == 'train_h5py':

        # add CAMELYON16 training dataset
        convert_to_h5py(cfg.path.train_dataset_patch, cfg.path.train_dataset_mask,
                        cfg.path.c16_patches_training, cfg.path.c16_patches_masks_training, False)

        # add CAMELYON16 testing dataset
        convert_to_h5py(cfg.path.train_dataset_patch, cfg.path.train_dataset_mask,
                        cfg.path.c16_patches_testing, cfg.path.c16_patches_masks_testing, False)

        # add CAMELYON17 training dataset
        convert_to_h5py(cfg.path.train_dataset_patch, cfg.path.train_dataset_mask,
                        cfg.path.c17_patches_training, cfg.path.c17_patches_masks_training, False)

    # create .csv file with patches and masks labels from CAMELYON16 training dataset and CAMELYON17 training dataset as the .csv file used for personal training
    if args.type == 'train_csv':

        # create list of available patches and their masks from CAMELYON16 training dataset in the right order
        c16_patches_list = [f for f in os.listdir(cfg.path.c16_patches_training) if path.isfile(path.join(cfg.path.c16_patches_training, f))]
        c16_masks_list = [patch.replace('patch', 'patch_mask') for patch in c16_patches_list]
        c16_patches_list = [cfg.path.c16_patches_training + s for s in c16_patches_list]
        c16_masks_list = [cfg.path.c16_patches_masks_training + s for s in c16_masks_list]

        # create list of available patches and their masks from CAMELYON17 training dataset in the right order
        c17_patches_list = [f for f in os.listdir(cfg.path.c17_patches_training) if path.isfile(path.join(cfg.path.c17_patches_training, f))]
        c17_masks_list = [patch.replace('patch', 'patch_mask') for patch in c17_patches_list]
        c17_patches_list = [cfg.path.c17_patches_training + s for s in c17_patches_list]
        c17_masks_list = [cfg.path.c17_patches_masks_training + s for s in c17_masks_list]

        # save list of available train patches to .csv file
        raw_data = {'patch': c16_patches_list + c17_patches_list, 'mask': c16_masks_list + c17_masks_list}
        df = pd.DataFrame(raw_data)
        df.to_csv(cfg.path.train_labels, header=True, index=False, sep=',')

    # create .csv file with patches and masks labels from CAMELYON16 testing dataset as the .csv file used for personal validation
    if args.type == 'val_csv':

        # create list of available patches and their masks from CAMELYON16 testing dataset in the right order
        c16_patches_list = [f for f in os.listdir(cfg.path.c16_patches_testing) if path.isfile(path.join(cfg.path.c16_patches_testing, f))]
        c16_masks_list = [patch.replace('patch', 'patch_mask') for patch in c16_patches_list]
        c16_patches_list = [cfg.path.c16_patches_testing + s for s in c16_patches_list]
        c16_masks_list = [cfg.path.c16_patches_masks_testing + s for s in c16_masks_list]

        # save list of available validation patches to .csv file
        raw_data = {'patch': c16_patches_list, 'mask': c16_masks_list}
        df = pd.DataFrame(raw_data)
        df.to_csv(cfg.path.val_labels, header=True, index=False, sep=',')

    # create .csv file with patches and masks labels of CAMELYON16 training, CAMELYON16 testing and CAMELYON17 training dataset as the general .csv file used for the final training
    if args.type == 'final_csv':

        # create list of available patches and their masks from CAMELYON16 training dataset in the right order
        c16_train_patches_list = [f for f in os.listdir(cfg.path.c16_patches_training) if path.isfile(path.join(cfg.path.c16_patches_training, f))]
        c16_train_masks_list = [patch.replace('patch', 'patch_mask') for patch in c16_train_patches_list]
        c16_train_patches_list = [cfg.path.c16_patches_training + s for s in c16_train_patches_list]
        c16_train_masks_list = [cfg.path.c16_patches_masks_training + s for s in c16_train_masks_list]

        # create list of available patches and their masks from CAMELYON16 testing dataset in the right order
        c16_test_patches_list = [f for f in os.listdir(cfg.path.c16_patches_testing) if path.isfile(path.join(cfg.path.c16_patches_testing, f))]
        c16_test_masks_list = [patch.replace('patch', 'patch_mask') for patch in c16_test_patches_list]
        c16_test_patches_list = [cfg.path.c16_patches_testing + s for s in c16_test_patches_list]
        c16_test_masks_list = [cfg.path.c16_patches_masks_testing + s for s in c16_test_masks_list]

        # create list of available patches and their masks from CAMELYON17 training dataset in the right order
        c17_train_patches_list = [f for f in os.listdir(cfg.path.c17_patches_training) if path.isfile(path.join(cfg.path.c17_patches_training, f))]
        c17_train_masks_list = [patch.replace('patch', 'patch_mask') for patch in c17_train_patches_list]
        c17_train_patches_list = [cfg.path.c17_patches_training + s for s in c17_train_patches_list]
        c17_train_masks_list = [cfg.path.c17_patches_masks_training + s for s in c17_train_masks_list]

        # save list of available test patches to .csv file
        raw_data = {'patch': c16_train_patches_list + c16_test_patches_list + c17_train_patches_list, 'mask': c16_train_masks_list + c16_test_masks_list + c17_train_masks_list}
        df = pd.DataFrame(raw_data)
        df.to_csv(cfg.path.final_labels, header=True, index=False, sep=',')

