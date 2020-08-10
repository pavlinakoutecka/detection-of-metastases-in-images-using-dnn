"""
File:           make_masks.py
Author:         Pavlina Koutecka
Date:           21/02/2020
Description:    This file generates various WSIs masks for visualization, preprocessing and patch extraction
                from arrays. For this purpose, this file contains also function for conversion
                .xml files to array of tumor coordinates.

                Masks, whose creation could be found in this file: slide mask, tumor region mask,
                non-tumor region mask, tissue region mask, slide with tumor boundaries mask.
"""

import cv2
import openslide
import numpy as np
import xml.etree.ElementTree as ET

import sys
import pathlib
parent_path = str(pathlib.Path().absolute().parent)
sys.path.append(parent_path)
import configuration as cfg


def xml_to_coordinates(slide_path, xml_path, mask_level):
    """
    This function converts given .xml file into array of tumor polygons. Each polygon contains
    array of (x,y) coordinates describing the tumor boundaries.

    :param slide_path: path to the .tif file of the slide
    :param xml_path: path to the .xml file of the slide
    :param mask_level: level to which should be the slide downsampled
    :return: array of tumor polygons with its coordinates
    """

    # initialize xml tree and polygon arrays
    tree = ET.parse(xml_path).getroot()
    polygon_list = []
    polygon_coordinates = []

    # initialize slide
    slide = openslide.OpenSlide(slide_path)  # open WSI
    level_factor = slide.level_downsamples[mask_level]  # get downsample factor of level mask_level (2**mask_level)

    # get the polygons coordinates
    for polygons in tree.iter('Coordinates'):
        for polygon in polygons:
            x_coord = float(polygon.get('X'))/level_factor  # downsample every x coordinate  by given factor
            y_coord = float(polygon.get('Y'))/level_factor  # downsample every y coordinate by given factor
            polygon_coordinates.append([round(x_coord), round(y_coord)])
        polygon_list.append(polygon_coordinates)
        polygon_coordinates = []

    return polygon_list


def make_slide_mask(slide_path, xml_path, slide_mask_path, mask_level):
    """
    This function converts slide in the .tif format to chosen image format with chosen
    downsampling level.

    :param slide_path: path to the .tif file of the slide
    :param xml_path: path to the .xml file of the slide
    :param slide_mask_path: path to the directory, where should be created slide mask stored
    :param mask_level: level to which should be the slide downsampled
    :return: mask (in OpenSlide format) of chosen slide
    """

    # initialize slide
    slide = openslide.OpenSlide(slide_path)  # open WSI
    width, height = slide.level_dimensions[mask_level]  # get WSIs dimensions of level mask_level

    # initialize slide mask - return an RGB Image containing the contents of the specified region
    slide_mask = slide.read_region((0, 0), mask_level, (width, height))
    #   cv2.read_region PARAMETERS:
    #       > location (tuple) – (x, y) tuple giving the top left pixel in the level 0 reference frame
    #       > level (int) – the level number
    #       > size (tuple) – (width, height) tuple giving the region size
    slide_mask = cv2.cvtColor(np.array(slide_mask), cv2.COLOR_RGBA2RGB)

    # draw tissue mask
    if slide_mask_path:
        cv2.imwrite(slide_mask_path, slide_mask)

    return slide_mask


def make_tumor_mask(slide_path, xml_path, tumor_mask_path, mask_level):
    """
    This function generates BW tumor mask. Tumor area is coloured white, rest of the slide
    is coloured black.

    :param slide_path: path to the .tif file of the slide
    :param xml_path: path to the .xml file of the slide
    :param tumor_mask_path: path to the directory, where should be created tumor mask stored
    :param mask_level: level to which should be the slide downsampled
    :return: tumor mask (in OpenSlide format) of chosen slide
    """

    # get tumor coordinates
    polygon_list = xml_to_coordinates(slide_path, xml_path, mask_level)

    # initialize slide
    slide = openslide.OpenSlide(slide_path)  # open WSI
    width, height = slide.level_dimensions[mask_level]  # get WSIs dimensions of level mask_level

    # initialize tumor mask
    tumor_mask = np.zeros((height, width))

    # draw tumor mask
    for polygon_coordinates in polygon_list:
        cv2.drawContours(tumor_mask, np.array([polygon_coordinates]), -1, (255, 255, 255), -1)
        #   cv2.drawContours PARAMETERS:
        #       > image – Destination image.
        #       > contours – All the input contours. Each contour is stored as a point vector.
        #       > contourIdx – Parameter indicating a contour to draw.
        #                      If it is negative, all the contours are drawn.
        #       > color – Color of the contours.
        #       > thickness – Thickness of lines the contours are drawn with.
        #                     If it is negative, the contour interiors are drawn.
        if tumor_mask_path:
            cv2.imwrite(tumor_mask_path, tumor_mask)

    return tumor_mask


def make_tumor_boundaries_mask(slide_path, xml_path, tumor_boundaries_mask_path, mask_level):
    """
    This function creates coloured boundaries around tumor areas on the slide.

    :param slide_path: path to the .tif file of the slide
    :param xml_path: path to the .xml file of the slide
    :param tumor_boundaries_mask_path: path to the directory, where should be created tumor boundaries mask stored
    :param mask_level: level to which should be the slide downsampled
    :return: tumor mask with boundaries (in OpenSlide format) of chosen slide
    """

    # get tumor coordinates
    polygon_list = xml_to_coordinates(slide_path, xml_path, mask_level)

    # initialize slide
    slide = openslide.OpenSlide(slide_path)  # open WSI
    width, height = slide.level_dimensions[mask_level]  # get WSIs dimensions of level mask_level

    # initialize slide mask - return an RGB image containing the contents of the specified region
    boundaries_mask = slide.read_region((0, 0), mask_level, (width, height))
    boundaries_mask = cv2.cvtColor(np.array(boundaries_mask), cv2.COLOR_RGBA2RGB)

    # draw boundary of tumor in slide map
    for polygon_coordinates in polygon_list:
        cv2.drawContours(boundaries_mask, np.array([polygon_coordinates]), -1, (255, 0, 0, 1), 10)
        if tumor_boundaries_mask_path:
            cv2.imwrite(tumor_boundaries_mask_path, boundaries_mask)

    return boundaries_mask


def make_tissue_region_mask(slide_path, xml_path, tissue_region_mask_path, mask_level):
    """
    This function identifies tissue regions on the slide and generates BW tissue mask. Tissue area
    is coloured white, rest of the slide is coloured black.
    Tissue regions are identified by Otsu's thresholding algorithm with additional morphological
    transformation (morphological closing) and median filtering.

    :param slide_path: path to the .tif file of the slide
    :param tissue_region_mask_path: path to the directory, where should be created tissue region mask stored
    :param mask_level: level to which should be the slide downsampled
    :return: tissue region mask (in OpenSlide format) of chosen slide
    """

    # initialize slide
    slide = openslide.OpenSlide(slide_path)  # open WSI
    width, height = slide.level_dimensions[mask_level]  # get WSIs dimensions of level mask_level

    # initialize slide mask - return an RGB image containing the contents of the specified region
    slide_mask = slide.read_region((0, 0), mask_level, (width, height))
    slide_mask = cv2.cvtColor(np.array(slide_mask), cv2.COLOR_RGBA2RGB)

    # get rid of black segments that might appear on the slide
    gray = cv2.cvtColor(slide_mask, cv2.COLOR_BGR2GRAY)
    gray_mask = cv2.compare(gray, 40, cv2.CMP_LT)
    slide_mask[gray_mask > 0] = 255

    # convert BGR to HSV
    slide_mask = cv2.cvtColor(slide_mask, cv2.COLOR_BGR2HSV)

    # generate threshold mask by Otsu's thresholding algorithm
    return_value, threshold_mask = cv2.threshold(slide_mask[:, :, 1], 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # morphological transformation
    #   taken from https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
    kernel = np.ones((5, 5))
    threshold_mask = cv2.morphologyEx(threshold_mask, cv2.MORPH_CLOSE, kernel)

    # median filter
    #   taken from https://docs.opencv.org/3.4/dc/dd3/tutorial_gausian_median_blur_bilateral_filter.html
    kernel = 5  # must be odd
    threshold_mask = cv2.medianBlur(threshold_mask, kernel)

    # draw tissue region mask
    if tissue_region_mask_path:
        cv2.imwrite(tissue_region_mask_path, threshold_mask)

    return threshold_mask


def make_normal_mask(slide_path, xml_path, normal_mask_path, mask_level):
    """
    This function generates BW non-tumor mask. Non-tumor tissue area is coloured white,
    rest of the slide is coloured black.

    :param slide_path: path to the .tif file of the slide
    :param xml_path: path to the .xml file of the slide
    :param normal_mask_path: path to the directory, where should be created normal mask stored
    :param mask_level: level to which should be the slide downsampled
    :return: non-tumor mask (in OpenSlide format) of chosen slide
    """

    # generate mask of non-tumor tissue region as subtraction of tumor and tissue region mask
    tissue_region_mask = make_tissue_region_mask(slide_path, xml_path, None, mask_level)
    tumor_mask = make_tumor_mask(slide_path, xml_path, None, mask_level)
    normal_mask = tissue_region_mask - tumor_mask

    # draw normal mask
    if normal_mask_path:
        cv2.imwrite(normal_mask_path, normal_mask)

    return normal_mask


"""
>>> EXAMPLES OF USAGE <<<

make_slide_mask(cfg.path.demo_slide, cfg.path.demo_xml, cfg.path.preprocessing_and_visualization + 'slide_mask.png', cfg.hyperparameter.mask_level)
make_tumor_mask(cfg.path.demo_slide, cfg.path.demo_xml, cfg.path.preprocessing_and_visualization + 'tumor_mask.png', cfg.hyperparameter.mask_level)
make_tumor_boundaries_mask(cfg.path.demo_slide, cfg.path.demo_xml, cfg.path.preprocessing_and_visualization + 'tumor_boundaries.png', cfg.hyperparameter.mask_level)
make_tissue_region_mask(cfg.path.demo_slide, cfg.path.demo_xml, cfg.path.preprocessing_and_visualization + 'tissue_region_mask.png', cfg.hyperparameter.mask_level)
make_normal_mask(cfg.path.demo_slide, cfg.path.demo_xml, cfg.path.preprocessing_and_visualization + 'normal_mask.png', cfg.hyperparameter.mask_level)
"""
