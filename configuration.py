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
    # =========
    mask_level = 4

    # patch extraction parameters
    patch_size = 256
    tumor_patches_per_slide = 500
    normal_patches_per_slide = int(tumor_patches_per_slide/2)

    # train parameters
    split_ratio = 1
    # =========
    num_classes = 2
    classes = ['No Tumor', 'Tumor']
    num_workers = 16
    epochs = 100
    batch_size = 1
    learning_rate = 5e-6

    # storage parameters
    environment = 'server'  # 'server' or 'cluster'
    save_run = False
    # =========
    date = now.strftime("%d_%m-%H_%M_%S")
    saving_string = f'__ps{str(patch_size)}__sr{str(split_ratio)}__{date}'
    saving_folder = '/mnt/datagrid/personal/koutepa2/bakprac/' if environment == 'server' else '/home/koutepa2/bakprac/'

    # net parameters
    model = 'deeplabv3_resnet101' #'deeplabv3_resnet101'
    trained_weights = saving_folder + f'Models/{model}/lvl_{str(patch_level)}/ep1__ps256__sr1__11_04-11_41_49.pt'
    model_title = 'DeepLabV3 with a ResNet-101 backbone'
    # =========
    pretrained = False


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
    extreme_patches = hyperparameter.saving_folder + 'Images/extreme_patches/'
    extreme_losses = hyperparameter.saving_folder + 'Images/extreme_losses/'
    images = hyperparameter.saving_folder + f'Images/{hyperparameter.model}/lvl_{str(hyperparameter.patch_level)}/'
    graphs = hyperparameter.saving_folder + f'Graphs/{hyperparameter.model}/lvl_{str(hyperparameter.patch_level)}/'
    models = hyperparameter.saving_folder + f'Models/{hyperparameter.model}/lvl_{str(hyperparameter.patch_level)}/'
    evaluate = hyperparameter.saving_folder + f'Evaluation/{hyperparameter.model}/lvl_{str(hyperparameter.patch_level)}/'

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
    demo_slide = source_folder + 'CAMELYON16/training/tumor/tumor_001.tif'  # slide used for testing
    demo_xml = source_folder + 'CAMELYON16/Annotations/tumor_001.xml'  # xml used for testing


class wsi:
    """

    """

    # TODO: normal_86 je potreba vynechat a presunout jako tumor_111 (napsat panu Heringovi)
    # TODO: test_049 je duplikovany snimek
    # TODO: test_114 neni fully annotated
    c16_error_train_slides = {'tumor_010.tif', 'tumor_015.tif', 'tumor_018.tif', 'tumor_020.tif', 'tumor_025.tif', 'tumor_029.tif', 'tumor_033.tif', 'tumor_034.tif', 'tumor_044.tif',
                              'tumor_046.tif', 'tumor_051.tif', 'tumor_054.tif', 'tumor_055.tif', 'tumor_056.tif', 'tumor_067.tif', 'tumor_079.tif', 'tumor_085.tif', 'tumor_092.tif',
                              'tumor_095.tif', 'tumor_110.tif', 'Normal_086.tif'}
    c16_error_test_slides = {'Test_114.tif', 'Test_049.tif', '.Copy of Test_039.tif.mufpVx'}

    # TODO: tyto snimky jsou oficialne dostupne, jen je musim pridat na datagrid (napsat panu Heringovi)
    c17_error_slides = {'patient_000_node_0.tif', 'patient_000_node_1.tif', 'patient_000_node_2.tif', 'patient_000_node_3.tif', 'patient_000_node_4.tif',
                        'patient_001_node_0.tif', 'patient_001_node_1.tif', 'patient_001_node_2.tif', 'patient_001_node_3.tif', 'patient_001_node_4.tif',
                        'patient_040_node_0.tif', 'patient_040_node_1.tif', 'patient_040_node_2.tif', 'patient_040_node_3.tif', 'patient_040_node_4.tif',
                        'patient_041_node_0.tif', 'patient_041_node_1.tif', 'patient_041_node_2.tif', 'patient_041_node_3.tif', 'patient_041_node_4.tif',
                        'patient_042_node_0.tif', 'patient_042_node_1.tif', 'patient_042_node_2.tif', 'patient_042_node_3.tif', 'patient_042_node_4.tif'}

