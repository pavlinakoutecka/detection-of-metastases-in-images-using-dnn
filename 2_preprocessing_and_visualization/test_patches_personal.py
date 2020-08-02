"""
FOR PERSONAL PURPOSES
"""


import cv2
import os
from os import path

import sys
import pathlib
parent_path = str(pathlib.Path().absolute().parent)
sys.path.append(parent_path)
import configuration as cfg
import utils


# # TESTUJ, ZDA EXISTUJE PRISLUSNA MASKA
# folder_patch = cfg.path.c16_patches_training
# folder_mask = cfg.path.c16_patches_masks_training
#
# slides_list = [f for f in os.listdir(folder_patch) if path.isfile(path.join(folder_patch, f))]
#
# for index, slide in enumerate(slides_list):
#
#     utils.visualize_progress(index, len(slides_list))
#
#     img = cv2.imread(folder_mask + slide.replace('patch', 'patch_mask'))
#
#     try:
#         typ = img.shape
#     except:
#         with open('missing_masks_c16_test' + '.txt', 'a') as f:
#             f.write(slide + "\n")
#         continue



# # TESTUJ, ZDA EXISTUJE PRISLUSNY PATCH
# folder_patch = cfg.path.c16_patches_training
# folder_mask = cfg.path.c16_patches_masks_training
#
# slides_list = [f for f in os.listdir(folder_mask) if path.isfile(path.join(folder_mask, f))]
#
# for index, slide in enumerate(slides_list):
#
#     utils.visualize_progress(index, len(slides_list))
#
#     img = cv2.imread(folder_patch + slide.replace('patch_mask', 'patch'))
#
#     try:
#         typ = img.shape
#     except:
#         with open('missing_patches_c16_test' + '.txt', 'a') as f:
#             f.write(slide + "\n")
#         continue
