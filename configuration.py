"""
File:           configuration.py
Author:         Pavlina Koutecka
Date:           21/02/2020
Description:    This file

"""


from datetime import datetime
now = datetime.now()


class hyperparameter:
    """

    """

    seed = 1

    # downsampling level parameters
    patch_level = 2
    mask_level = 5

    # patch extraction parameters
    patch_size = 256
    tumor_patches_per_slide = 500
    normal_patches_per_slide = int(tumor_patches_per_slide/2)
    patches_samples = 100

    # train parameters
    split_ratio = 10
    # =========
    num_classes = 2
    classes = ['No Tumor', 'Tumor']
    num_workers = 16
    epochs = 100
    batch_size = 16  # training ... 16, evaluating ... 128
    learning_rate = 5e-6

    # storage parameters
    environment = 'server'  # 'server' or 'cluster'
    save_run = False
    # =========
    date = now.strftime("%d_%m-%H_%M_%S")
    saving_string = f'__ps{str(patch_size)}__sr{str(split_ratio)}__{date}'
    saving_folder = '/mnt/datagrid/personal/koutepa2/bakprac/' if environment == 'server' else '/home/koutepa2/bakprac/'

    # net parameters
    model = 'unet_resnet50'  # 'deeplabv3_resnet101' or 'fcn_resnet50' or 'unet_resnet50'
    trained_weights = False
    # trained_weights = saving_folder + f'Models/{model}/lvl_{str(patch_level)}/ep2__ps256__sr1__11_04-11_41_49.pt'
    #   level 1: ep0__ps256__sr1__10_04-11_35_35.pt
    #   level 2: ep1__ps256__sr1__11_04-11_41_49.pt, ep2__ps256__sr1__11_04-11_41_49.pt
    model_title = 'DeepLabV3 with a ResNet-101 backbone'
    # =========
    pretrained = False

    # evaluation parameters
    threshold_area = 40
    #   level 1: 40 (thresholded for 254)
    #   level 2: 66 (thresholded for 254)
    threshold_saving_string = 'grid_lvl_1_threshold_254.txt'


class path(hyperparameter):
    """

    """
    # main source folder
    source_folder = '/mnt/medical/microscopy/' if hyperparameter.environment == 'server' else '/home/koutepa2/bakprac/'

    # CAMELYON16 source folders
    c16_training_normal = source_folder + 'CAMELYON16/training/normal/'
    c16_training_tumor = source_folder + 'CAMELYON16/training/tumor/'
    c16_testing = source_folder + 'CAMELYON16/testing/'
    c16_annotations = source_folder + 'CAMELYON16/Annotations/'  # tumor and test annotations

    # CAMELYON17 source folders
    c17_training = source_folder + 'CAMELYON17/training/'
    c17_testing = source_folder + 'CAMELYON17/testing/'
    c17_annotations = source_folder + 'CAMELYON17/annotations/'  # tumor annotations
    c17_stage_labels = source_folder + 'CAMELYON17/supplemental/stage_labels.csv'

    # =============================================================================================================

    # section folders
    preparation = hyperparameter.saving_folder + '1_preparation/'
    preprocessing_and_visualization = hyperparameter.saving_folder + '2_preprocessing_and_visualization/'
    slide_level_classification = hyperparameter.saving_folder + '3_slide_level_classification/'
    patient_level_classification = hyperparameter.saving_folder + '4_patient_level_classification/'

    # images, graphs and models folders
    extreme_patches = hyperparameter.saving_folder + f'Images/{hyperparameter.model}/lvl_{str(hyperparameter.patch_level)}/extreme_patches/'
    extreme_losses = hyperparameter.saving_folder + f'Images/{hyperparameter.model}/lvl_{str(hyperparameter.patch_level)}/extreme_losses/'
    patches_example = hyperparameter.saving_folder + f'Images/{hyperparameter.model}/lvl_{str(hyperparameter.patch_level)}/patches_example/'
    images = hyperparameter.saving_folder + f'Images/{hyperparameter.model}/lvl_{str(hyperparameter.patch_level)}/'
    graphs = hyperparameter.saving_folder + f'Graphs/{hyperparameter.model}/lvl_{str(hyperparameter.patch_level)}/'
    models = hyperparameter.saving_folder + f'Models/{hyperparameter.model}/lvl_{str(hyperparameter.patch_level)}/'
    evaluate_slide = hyperparameter.saving_folder + f'3_slide_level_classification/Evaluation/{hyperparameter.model}/lvl_{str(hyperparameter.patch_level)}/'
    evaluate_patient = hyperparameter.saving_folder + f'4_patient_level_classification/Evaluation/{hyperparameter.model}/lvl_{str(hyperparameter.patch_level)}/'

    # personal CAMELYON16 folders
    c16_patches_training = hyperparameter.saving_folder + f'CAMELYON16_personal/patches_training_lvl_{str(hyperparameter.patch_level)}/'
    c16_patches_testing = hyperparameter.saving_folder + f'CAMELYON16_personal/patches_testing_lvl_{str(hyperparameter.patch_level)}/'
    c16_patches_masks_training = hyperparameter.saving_folder + f'CAMELYON16_personal/patches_masks_training_lvl_{str(hyperparameter.patch_level)}/'
    c16_patches_masks_testing = hyperparameter.saving_folder + f'CAMELYON16_personal/patches_masks_testing_lvl_{str(hyperparameter.patch_level)}/'
    c16_masks_training = hyperparameter.saving_folder + 'CAMELYON16_personal/masks_training/'
    c16_masks_testing = hyperparameter.saving_folder + 'CAMELYON16_personal/masks_testing/'

    # personal CAMELYON17 folders
    c17_patches_training = hyperparameter.saving_folder + f'CAMELYON17_personal/patches_training_lvl_{str(hyperparameter.patch_level)}/'
    c17_patches_masks_training = hyperparameter.saving_folder + f'CAMELYON17_personal/patches_masks_training_lvl_{str(hyperparameter.patch_level)}/'
    c17_masks_training = hyperparameter.saving_folder + 'CAMELYON17_personal/masks_training/'
    c17_masks_testing = hyperparameter.saving_folder + 'CAMELYON17_personal/masks_testing/'

    # datasets and labels path
    train_dataset_patch = hyperparameter.saving_folder + f'Datasets/train_dataset_patch_lvl_{str(hyperparameter.patch_level)}.h5'
    train_dataset_mask = hyperparameter.saving_folder + f'Datasets/train_dataset_mask_lvl_{str(hyperparameter.patch_level)}.h5'
    train_labels = hyperparameter.saving_folder + f'Datasets/train_labels_lvl_{str(hyperparameter.patch_level)}.csv'
    val_labels = hyperparameter.saving_folder + f'Datasets/val_labels_lvl_{str(hyperparameter.patch_level)}.csv'
    final_labels = hyperparameter.saving_folder + f'Datasets/labels_lvl_{str(hyperparameter.patch_level)}.csv'

    # demo slide path
    demo_slide = source_folder + 'CAMELYON16/testing/Test_010.tif'  # slide used for testing
    demo_xml = source_folder + 'CAMELYON16/Annotations/test_010.xml'  # xml used for testing


class wsi:
    """

    """

    # Test_049: duplicated slide, other slides: annotations are not exhaustive
    c16_error_slides = {'tumor_010.tif', 'tumor_015.tif', 'tumor_018.tif', 'tumor_020.tif', 'tumor_025.tif', 'tumor_029.tif', 'tumor_033.tif', 'tumor_034.tif', 'tumor_044.tif',
                        'tumor_046.tif', 'tumor_051.tif', 'tumor_054.tif', 'tumor_055.tif', 'tumor_056.tif', 'tumor_067.tif', 'tumor_079.tif', 'tumor_085.tif',
                        'tumor_092.tif', 'tumor_095.tif', 'tumor_110.tif', 'Test_114.tif', 'Test_049.tif'}
    c17_error_slides = {}

