"""
File:           train_network.py
Author:         Pavlina Koutecka
Date:           09/03/2020
Description:    This file aims to train and validate specified model. This process could be repeated for chosen
                number of epochs. You can choose, what process you want to do:

                    > train and validate the model - this function runs the process of training and validating the model -
                    trains and validates the model for given number of epochs, prints and saves its stats, saves trained
                    model, saves loss and accuracy curve; it also reduces LR when loss has stopped decreasing
                    > evaluate the model - this function runs the process of evaluating the model - computes one epoch,
                    prints its stats and creates confusion matrix
                    > find learning rate - this function runs process of searching for optimal learning rate of chosen model
                    > find random samples - this function stores to .png defined number of patches from validation dataset
                    > find extremal samples - this function finds patches from validation dataset with lowest and highest loss.
"""


import numpy as np
import pandas as pd
import random
import time
import math
import cv2
from datetime import datetime
now = datetime.now()
import matplotlib.pyplot as plt
import albumentations
from albumentations.pytorch import ToTensor
import segmentation_models_pytorch as smp

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional
from torch.autograd import Variable

import sys
import pathlib
parent_path = str(pathlib.Path().absolute().parent)
sys.path.append(parent_path)
import configuration as cfg
import train_utils
import plot_utils
import utils

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


def load_data():
    """
    This function converts prepared training and validation datasets into required format that is
    used for training and validation.

    :return: training and validation loader
    """

    # load dataset in required format
    train_labels = pd.read_csv(cfg.path.train_labels, error_bad_lines=False)
    val_labels = pd.read_csv(cfg.path.val_labels, error_bad_lines=False)
    train_dataset = train_utils.TrainDataset(train_labels, train_transforms)
    val_dataset = train_utils.TrainDataset(val_labels, val_transforms)

    # cut dataset to learn faster (just for exploration, use split_ratio = 1 for full dataset)
    train_split_len = len(train_dataset) // cfg.hyperparameter.split_ratio
    val_split_len = len(val_dataset) // cfg.hyperparameter.split_ratio
    part_train = torch.utils.data.random_split(train_dataset, [train_split_len, len(train_dataset) - train_split_len])[0]
    part_val = torch.utils.data.random_split(val_dataset, [val_split_len, len(val_dataset) - val_split_len])[0]

    # convert dataset into DataLoader format to be read properly
    loader_train = DataLoader(part_train, batch_size=cfg.hyperparameter.batch_size, shuffle=True, num_workers=cfg.hyperparameter.num_workers)
    loader_val = DataLoader(part_val, batch_size=cfg.hyperparameter.batch_size, shuffle=False, num_workers=cfg.hyperparameter.num_workers)

    # print number of training and validation patches
    print(f"\nBefore cut: There are {len(train_dataset)} patches in train.")
    print(f"Before cut: There are {len(val_dataset)} patches in val.")
    print(f"After cut: there are {len(part_train)} patches in train.")
    print(f"After cut: there are {len(part_val)} patches in val.\n")

    return loader_train, loader_val


def find_lr(trn_loader, init_value=1e-12, final_value=10.):
    """
    Simple implementation of the learning rate range test. This test can provide information about optimal
    learning rate. During the process, we start with small learning rate which is increased exponentially
    after each mini-batch between two values (init_value and final_value) or till loss starts exploding.
    Increasing learning rate allows network's loss to start to converge and as it grows, the loss will eventually
    diverge. A good learning rate for starting the training after this test would be half-way on the descending loss curve.

    This method was proposed by Leslie N. Smith in 2015 and updated in 2017.

    :param trn_loader: data loader that loads the data into batches as input for the model
    :param init_value: initial learning rate value
    :param final_value: maximum learning rate value to try
    :return: None

    References:
        https://arxiv.org/abs/1803.09820
        https://arxiv.org/abs/1506.01186
        https://sgugger.github.io/how-do-you-find-a-good-learning-rate.html
        https://gist.github.com/karanchahal/dc0575bf21b976ea633dea8ceffaf9dc

    Warning: For correct testing turn off loading pretrained weights of the model!
    """

    model.train()  # setup model for training configuration

    num = len(trn_loader)-1  # total number of batches
    mult = (final_value / init_value) ** (1/num)

    losses = []
    log_lrs = []
    best_loss = 0.
    avg_loss = 0.
    beta = 0.98  # the value for smooth losses
    lr = init_value

    for batch_num, data in enumerate(trn_loader):

        optimizer.param_groups[0]['lr'] = lr
        batch_num += 1  # for non zero value

        # prepare the mini-batch
        inputs, labels = data
        inputs, labels = Variable(inputs), Variable(labels)
        inputs, labels = inputs.to(device), labels.to(device)

        # get the loss for this mini-batch
        optimizer.zero_grad()
        outputs = model(inputs)['out']
        loss = criterion(outputs, labels)

        # compute the smoothed loss to create a clean graph
        avg_loss = beta * avg_loss + (1-beta) * loss.data
        smoothed_loss = avg_loss / (1 - beta**batch_num)

        # record the best loss
        if smoothed_loss < best_loss or batch_num == 1:
            best_loss = smoothed_loss

        # stop if the loss is exploding
        if batch_num > 1 and smoothed_loss > 8 * best_loss:
            break

        # append loss and learning rates for plotting
        losses.append(smoothed_loss)
        log_lrs.append(math.log10(lr))

        # backpropagate for the next step
        loss.backward()
        optimizer.step()

        # update the LR for the next step
        lr *= mult
        optimizer.param_groups[0]['lr'] = lr

        # print loss info after one episode of searching
        print('Actual lr: {:.9f}\t Actual loss: {:.4f}'.format(lr, smoothed_loss), end='\r')

    # plot results of finding optimal LR
    plt.plot(log_lrs[10:-5], losses[10:-5], "b-")
    plt.xlabel("Learning rate")
    plt.ylabel("Loss")
    plt.title("Learning rate searching process for " + cfg.hyperparameter.model_title)
    plt.savefig(cfg.path.graphs + f'lr_searcher' + cfg.hyperparameter.saving_string + '.pdf', format='pdf')
    plt.show()


def find_random_samples():
    """
    This function stores defined number of randomly chosen patches from validation dataset (number of samples
    is defined by the variable patches_samples in the configuration.py file). These samples are stored to
    .png files with its true mask and predicted mask.

    :return: None

    Warning: For running this function correctly, change the batch size in configuration file to 1!
    """

    patches_array = np.empty((cfg.hyperparameter.patches_samples, cfg.hyperparameter.patch_size, cfg.hyperparameter.patch_size, 3), dtype=float)
    masks_array = np.empty((cfg.hyperparameter.patches_samples, cfg.hyperparameter.patch_size, cfg.hyperparameter.patch_size), dtype=float)
    predictions_array = np.empty((cfg.hyperparameter.patches_samples, cfg.hyperparameter.patch_size, cfg.hyperparameter.patch_size), dtype=float)

    # turn the model to validation mode
    model.eval()

    # start and compute estimated time till the end of the epoch
    start_time = time.time()
    remaining_seconds = None

    # disable gradient computation
    with torch.no_grad():

        # validate epoch in mini-batches
        for batch_idx, data in enumerate(val_loader):

            if batch_idx >= cfg.hyperparameter.patches_samples:
                break

            # load patches and its masks from validation loader
            patches, masks = data
            patches = patches.to(device)
            masks = masks.to(device)

            # forward pass
            outputs = model(patches)['out']

            # apply softmax function
            outputs = torch.nn.functional.softmax(outputs, dim=1)

            patch = 255 * patches[0].cpu().numpy()
            patch = np.transpose(patch, (1, 2, 0))
            patches_array[batch_idx] = patch
            masks_array[batch_idx] = 255 * masks[0].cpu().numpy()
            predictions_array[batch_idx] = 255 * outputs[0][1].cpu().numpy()

            # compute estimated time till the end of epoch
            if batch_idx % 10 == 0 and batch_idx != 0:
                stop_time = time.time()
                remaining_seconds = (stop_time - start_time) * (len(val_loader) - batch_idx) / 10
                start_time = time.time()

            # print info
            utils.visualize_progress(batch_idx, len(val_loader), remaining_seconds)

    # store patches with lowest and highest losses to .png files
    for i in range(cfg.hyperparameter.patches_samples):
        cv2.imwrite(cfg.path.patches_example + f'sample_patch{i}.png', patches_array[i])
        cv2.imwrite(cfg.path.patches_example + f'sample_mask{i}.png', masks_array[i])
        cv2.imwrite(cfg.path.patches_example + f'sample_prediction{i}.png', predictions_array[i])


def find_extremal_samples():
    """
    This function finds patches from validation dataset with lowest and highest loss. Arrays for storing patches with
    lowest and highest loss are prepared. After that, patches are loaded into trained model and if their loss is bigger
    or smaller than any of the element in prepared arrays, this element is replaced with loaded patch. After this
    process, chosen number of extreme examples (patches) is stored to .png files with its true mask and predicted mask.

    :return: None

    Warning: For running this function correctly, change the batch size in configuration file to 1!
    """

    # prepare all the lowest loss necessities needed for finding patches, their true masks and predictions and storing them
    lowest_loss = np.empty(cfg.hyperparameter.patches_samples)
    lowest_loss.fill(math.inf)  # at the start, fill the array with infinity values --> we are trying to minimize
    lowest_loss_patch = np.empty((cfg.hyperparameter.patches_samples, cfg.hyperparameter.patch_size, cfg.hyperparameter.patch_size, 3), dtype=float)
    lowest_loss_mask = np.empty((cfg.hyperparameter.patches_samples, cfg.hyperparameter.patch_size, cfg.hyperparameter.patch_size), dtype=float)
    lowest_loss_prediction = np.empty((cfg.hyperparameter.patches_samples, cfg.hyperparameter.patch_size, cfg.hyperparameter.patch_size), dtype=float)

    # prepare all the highest loss necessities needed for finding patches, their true masks and predictions and storing them
    highest_loss = np.zeros(cfg.hyperparameter.patches_samples)  # at the start, fill the array with zero values --> we are trying to maximize
    highest_loss_patch = np.empty((cfg.hyperparameter.patches_samples, cfg.hyperparameter.patch_size, cfg.hyperparameter.patch_size, 3), dtype=float)
    highest_loss_mask = np.empty((cfg.hyperparameter.patches_samples, cfg.hyperparameter.patch_size, cfg.hyperparameter.patch_size), dtype=float)
    highest_loss_prediction = np.empty((cfg.hyperparameter.patches_samples, cfg.hyperparameter.patch_size, cfg.hyperparameter.patch_size), dtype=float)

    # turn the model to validation mode
    model.eval()

    # start and compute estimated time till the end of the epoch
    start_time = time.time()
    remaining_seconds = None

    # disable gradient computation
    with torch.no_grad():

        # validate epoch in mini-batches
        for batch_idx, data in enumerate(val_loader):

            # load patches and its masks from validation loader
            patches, masks = data
            patches = patches.to(device)
            masks = masks.to(device)

            # forward pass
            outputs = model(patches)['out']

            # calculate loss and apply softmax function
            loss = criterion(outputs, masks)
            outputs = torch.nn.functional.softmax(outputs, dim=1)

            # find indices whose loss value is smaller (in the case of lowest loss) or bigger (in the case of highest loss) than values stored in the array
            low_indices = [i for i, v in enumerate(lowest_loss) if v > loss]
            high_indices = [i for i, v in enumerate(highest_loss) if v < loss]

            # if we found patch in the array whose loss is bigger than calculated loss, replace it with this new patch
            if low_indices:
                lowest_loss[low_indices[0]] = loss
                patch = 255 * patches[0].cpu().numpy()
                patch = np.transpose(patch, (1, 2, 0))
                lowest_loss_patch[low_indices[0]] = patch
                lowest_loss_mask[low_indices[0]] = 255 * masks[0].cpu().numpy()
                lowest_loss_prediction[low_indices[0]] = 255 * outputs[0][1].cpu().numpy()

            # if we found patch in the array whose loss is smaller than calculated loss, replace it with this new patch
            if high_indices:
                highest_loss[high_indices[0]] = loss
                patch = 255 * patches[0].cpu().numpy()
                patch = np.transpose(patch, (1, 2, 0))
                highest_loss_patch[high_indices[0]] = patch
                highest_loss_mask[high_indices[0]] = 255 * masks[0].cpu().numpy()
                highest_loss_prediction[high_indices[0]] = 255 * outputs[0][1].cpu().numpy()

            # compute estimated time till the end of epoch
            if batch_idx % 10 == 0 and batch_idx != 0:
                stop_time = time.time()
                remaining_seconds = (stop_time - start_time) * (len(val_loader) - batch_idx) / 10
                start_time = time.time()

            # print info
            utils.visualize_progress(batch_idx, len(val_loader), remaining_seconds)

    # store patches with lowest and highest losses to .png files
    for i in range(len(lowest_loss_patch)):
        cv2.imwrite(cfg.path.extreme_losses + f'lowest_loss_patch{i}.png', lowest_loss_patch[i])
        cv2.imwrite(cfg.path.extreme_losses + f'lowest_loss_mask{i}.png', lowest_loss_mask[i])
        cv2.imwrite(cfg.path.extreme_losses + f'lowest_loss_prediction{i}.png', lowest_loss_prediction[i])
    for i in range(len(highest_loss_patch)):
        cv2.imwrite(cfg.path.extreme_losses + f'highest_loss_patch{i}.png', highest_loss_patch[i])
        cv2.imwrite(cfg.path.extreme_losses + f'highest_loss_mask{i}.png', highest_loss_mask[i])
        cv2.imwrite(cfg.path.extreme_losses + f'highest_loss_prediction{i}.png', highest_loss_prediction[i])


def train_epoch(epoch):
    """
    This function is main function used for training the model.

    :param epoch: number of current epoch
    :return: average loss, accuracy, sensitivity, specificity, precision, recall, iou,
             dice_coefficient, confusion_matrix
    """

    # turn the model to training mode
    model.train()

    # prepare loss object
    losses = train_utils.Average()

    # start and compute estimated time till the end of the epoch
    start_time = time.time()
    remaining_seconds = None

    # train epoch in mini-batches
    for batch_idx, data in enumerate(train_loader):

        # load patches and its masks from train loader
        patches, masks = data
        patches = patches.to(device)
        masks = masks.to(device)

        # if batch_idx == 0:
        #     img = patches[3].permute(1,2,0).cpu().numpy()
        #     plt.imsave("aug_hsv.png", img)

        # zero gradients
        optimizer.zero_grad()

        # forward pass
        outputs = model(patches)

        # calculate loss
        loss = criterion(outputs, masks)
        losses.update(loss.data.cpu().numpy(), masks.size(0))

        # backward pass
        loss.backward()

        # optimize
        optimizer.step()

        # calculate metrics (accuracy, precision, recall, iou, dice coefficient, confusion matrix)
        metrics.update(outputs.detach(), masks.detach())

        # plot train loss and accuracy after bunch of mini-batches have finished
        if (batch_idx*cfg.hyperparameter.batch_size) % int(len(train_loader)/20) == 0:
            accuracy, sensitivity, specificity, precision, recall, iou, dice_coefficient, confusion_matrix = metrics.compute()
            train_accuracy_global.append(accuracy)
            train_loss_global.append(losses.avg)
            train_axis.append(epoch+batch_idx/len(train_loader))

        # compute estimated time till the end of epoch
        if batch_idx % 10 == 0 and batch_idx != 0:
            stop_time = time.time()
            remaining_seconds = (stop_time - start_time) * (len(train_loader) - batch_idx) / 10
            start_time = time.time()

        # print info
        text = " TRAIN | Epoch: {}/{}".format(epoch + 1, cfg.hyperparameter.epochs)
        utils.visualize_progress(batch_idx, len(train_loader), remaining_seconds, text=text)

    # compute and return stats
    accuracy, sensitivity, specificity, precision, recall, iou, dice_coefficient, confusion_matrix = metrics.compute()
    return losses.avg, accuracy, sensitivity, specificity, precision, recall, iou, dice_coefficient, confusion_matrix


def validate_epoch(epoch):
    """
    This function is main function used for validating the model.

    :param epoch: number of current epoch
    :return: average loss, accuracy, sensitivity, specificity, precision, recall, iou,
             dice_coefficient, confusion_matrix
    """

    # turn the model to validation mode
    model.eval()

    # prepare loss object
    losses = train_utils.Average()

    # start and compute estimated time till the end of the epoch
    start_time = time.time()
    remaining_seconds = None

    # disable gradient computation
    with torch.no_grad():

        # validate epoch in mini-batches
        for batch_idx, data in enumerate(val_loader):

            # load patches and its masks from validation loader
            patches, masks = data
            patches = patches.to(device)
            masks = masks.to(device)

            # forward pass
            outputs = model(patches)['out']

            # calculate loss
            loss = criterion(outputs, masks)
            losses.update(loss.data.cpu().numpy(), masks.size(0))

            # calculate metrics (accuracy, precision, recall, iou, dice coefficient, confusion matrix)
            metrics.update(outputs, masks)

            # plot validation loss and accuracy after bunch of mini-batches have finished
            if (batch_idx*cfg.hyperparameter.batch_size) % int(len(val_loader)/5) == 0:
                accuracy, sensitivity, specificity, precision, recall, iou, dice_coefficient, confusion_matrix = metrics.compute()
                val_accuracy_global.append(accuracy)
                val_loss_global.append(losses.avg)
                val_axis.append(epoch + batch_idx / len(val_loader))

            # compute estimated time till the end of epoch
            if batch_idx % 10 == 0 and batch_idx != 0:
                stop_time = time.time()
                remaining_seconds = (stop_time - start_time) * (len(val_loader) - batch_idx) / 10
                start_time = time.time()

            # print info
            text = " VAL | Epoch: {}/{}".format(epoch + 1, cfg.hyperparameter.epochs)
            utils.visualize_progress(batch_idx, len(val_loader), remaining_seconds, text=text)

    # compute and return stats
    accuracy, sensitivity, specificity, precision, recall, iou, dice_coefficient, confusion_matrix = metrics.compute()
    return losses.avg, accuracy, sensitivity, specificity, precision, recall, iou, dice_coefficient, confusion_matrix


def process_of_training():
    """
    This function runs the process of training and validating the model - trains and validates the model for given number
    of epochs, prints and saves its stats, saves trained model, saves loss and accuracy curve. It also reduces LR when
    loss has stopped decreasing.

    :return: None
    """

    # save training info as .txt file
    if cfg.hyperparameter.save_run:
        with open(cfg.path.models + f'training_history' + cfg.hyperparameter.saving_string + '.txt', 'a') as f:
            f.write(f'\n--- Date: {cfg.hyperparameter.date}\n'
                    f'--- Type: {args.type}\n'
                    f'--- Save this run: {cfg.hyperparameter.save_run}\n'
                    f'--- Model: {cfg.hyperparameter.model}\n'
                    f'--- Pretrained: {cfg.hyperparameter.pretrained}\n'
                    f'--- Loaded weights: {cfg.hyperparameter.trained_weights}\n'
                    f'--- Split ratio: {cfg.hyperparameter.split_ratio}\n'
                    f'--- Patch level: {cfg.hyperparameter.patch_level}\n'
                    f'--- Mask level: {cfg.hyperparameter.mask_level}\n'
                    f'--- Patch size: {cfg.hyperparameter.patch_size}\n'
                    f'--- Batch size: {cfg.hyperparameter.batch_size}\n'
                    f'--- Learning rate: {cfg.hyperparameter.learning_rate}\n')

    # values to store best train and validation accuracy
    best_accuracy_train = float(0)
    best_accuracy_val = float(0)

    # train the model for given number of epochs
    for epoch in range(cfg.hyperparameter.epochs):

        # compute a training epoch, after that reset metrics
        train_loss, train_accuracy, train_sensitivity, train_specificity, train_precision, train_recall, train_iou, train_dice_coefficient, confusion_matrix = train_epoch(epoch)
        metrics.reset()

        # compute a validation epoch, after that reset metrics
        val_loss, val_accuracy, val_sensitivity, val_specificity, val_precision, val_recall, val_iou, val_dice_coefficient, confusion_matrix = validate_epoch(epoch)
        metrics.reset()

        # reduce LR when a metric (loss) has stopped decreasing
        scheduler.step(val_loss)

        # print training and validation info
        best_accuracy_train = max(train_accuracy, best_accuracy_train)
        best_accuracy_val = max(val_accuracy, best_accuracy_val)
        print("\n\t>>> TRAIN\t accuracy: {:.5f} (best) - {:.5f} (current)\t  loss: {:.5f}".format(best_accuracy_train, train_accuracy, train_loss))
        print("\t>>> VALIDATION\t accuracy: {:.5f} (best) - {:.5f} (current)\t  loss: {:.5f}\n".format(best_accuracy_val, val_accuracy, val_loss))

        # save all the training stuffs
        if cfg.hyperparameter.save_run:

            # save models checkpoint (weights) as .pt file
            torch.save(model.state_dict(), cfg.path.models + f'ep{str(epoch)}' + cfg.hyperparameter.saving_string + '.pt')

            # save training history as .txt file
            with open(cfg.path.models + f'training_history' + cfg.hyperparameter.saving_string + '.txt', 'a') as f:
                f.write("\nEPOCH: {}/{}\n".format(epoch + 1, cfg.hyperparameter.epochs))
                f.write("\t>>> TRAIN\t accuracy: {:.5f} (best) - {:.5f} (current)\t  loss: {:.5f}\n".format(best_accuracy_train, train_accuracy, train_loss))
                f.write("\t>>> VALIDATION\t accuracy: {:.5f} (best) - {:.5f} (current)\t  loss: {:.5f}\n".format(best_accuracy_val, val_accuracy, val_loss))

            # save loss curve to .csv file
            df = pd.DataFrame({'axis': train_axis, 'loss': train_loss_global})
            df.to_csv(cfg.path.graphs + f'ep{str(epoch)}__train_loss' + cfg.hyperparameter.saving_string + '.csv', header=False, index=False)
            df = pd.DataFrame({'axis': val_axis, 'loss': val_loss_global})
            df.to_csv(cfg.path.graphs + f'ep{str(epoch)}__val_loss' + cfg.hyperparameter.saving_string + '.csv', header=False, index=False)

            # plot loss curve to .png file
            plt.figure()
            plt.plot(train_axis, train_loss_global, "b-", label='training loss')
            plt.plot(val_axis, val_loss_global, "r-", label='validation loss')
            plt.xlabel("epoch")
            plt.ylabel("loss")
            plt.title("Model loss")
            plt.legend()
            plt.savefig(cfg.path.graphs + f'ep{str(epoch)}__model_loss' + cfg.hyperparameter.saving_string + '.png')
            plt.show()

            # save accuracy curve to .csv file
            df = pd.DataFrame({'axis': train_axis, 'acc': train_accuracy_global})
            df.to_csv(cfg.path.graphs + f'ep{str(epoch)}__train_accuracy' + cfg.hyperparameter.saving_string + '.csv', header=False, index=False)
            df = pd.DataFrame({'axis': val_axis, 'acc': val_accuracy_global})
            df.to_csv(cfg.path.graphs + f'ep{str(epoch)}__val_accuracy' + cfg.hyperparameter.saving_string + '.csv', header=False, index=False)

            # plot accuracy curve to .png file
            plt.figure()
            plt.plot(train_axis, train_accuracy_global, "b-", label='training accuracy')
            plt.plot(val_axis, val_accuracy_global, "r-", label='validation accuracy')
            plt.xlabel("epoch")
            plt.ylabel("accuracy")
            plt.title("Model accuracy")
            plt.legend()
            plt.savefig(cfg.path.graphs + f'ep{str(epoch)}__model_accuracy' + cfg.hyperparameter.saving_string + '.png')
            plt.show()

            # plot smoothed accuracy and loss curves to .png file
            plot_utils.plot_smoothed_curve(cfg.path.graphs + f'ep{str(epoch)}__smoothed' + cfg.hyperparameter.saving_string + '.png',
                                           cfg.path.graphs + f'ep{str(epoch)}__train_accuracy' + cfg.hyperparameter.saving_string + '.csv',
                                           cfg.path.graphs + f'ep{str(epoch)}__val_accuracy' + cfg.hyperparameter.saving_string + '.csv',
                                           cfg.path.graphs + f'ep{str(epoch)}__train_loss' + cfg.hyperparameter.saving_string + '.csv',
                                           cfg.path.graphs + f'ep{str(epoch)}__val_loss' + cfg.hyperparameter.saving_string + '.csv')


def process_of_evaluation():
    """
    This function runs the process of evaluating the model - computes one epoch, prints its stats and
    creates confusion matrix.

    :return: None
    """

    # compute an evaluation epoch
    loss, accuracy, sensitivity, specificity, precision, recall, iou, dice_coefficient, confusion_matrix = validate_epoch(epoch=0)

    # print evaluation info
    print("\n ACCURACY: {}\t LOSS: {}\n".format(accuracy, loss))
    metrics.print()

    # plot confusion matrix to .png file
    if cfg.hyperparameter.save_run:
        plot_utils.plot_confusion_matrix(cm=confusion_matrix, normalize=False, path=cfg.path.graphs + f'confusion_matrix' + cfg.hyperparameter.saving_string + '.png',
                                         title="Confusion matrix", cmap=None, target_names=cfg.hyperparameter.classes)


if __name__ == "__main__":

    args = utils.parse_args()

    # load train and validation transforms
    train_transforms = albumentations.Compose([
        albumentations.augmentations.transforms.Flip(always_apply=False, p=0.5),
        albumentations.augmentations.transforms.RandomRotate90(always_apply=False, p=0.5),
        albumentations.augmentations.transforms.RandomBrightness(limit=0.2, always_apply=False, p=0.5),
        albumentations.augmentations.transforms.RandomContrast(limit=0.2, always_apply=False, p=0.5),
        albumentations.augmentations.transforms.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, always_apply=False, p=0.5),
        ToTensor(),
    ])
    val_transforms = albumentations.Compose([
        ToTensor(),
    ])

    # load the dataset in required format
    print("Loading training data...")
    train_loader, val_loader = load_data()

    # load the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # '.to(device)' can be written also as '.cuda()'
    # load the model
    if cfg.hyperparameter.model == 'deeplabv3_resnet101':
        model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=cfg.hyperparameter.pretrained, progress=True, num_classes=cfg.hyperparameter.num_classes)
    if cfg.hyperparameter.model == 'fcn_resnet50':
        model = torchvision.models.segmentation.fcn_resnet50(pretrained=cfg.hyperparameter.pretrained, progress=True, num_classes=cfg.hyperparameter.num_classes)
    if cfg.hyperparameter.model == 'unet_resnet50':
        model = smp.Unet('resnet50', classes=cfg.hyperparameter.num_classes)
    model.to(device)

    # replace the last convolutional layer of backbone classifier (21 output classes transformed into 2 output classes) if the model is pretrained
    if cfg.hyperparameter.pretrained:
        for param in model.parameters():
            param.requires_grad = False
        # parameters of newly constructed layers have requires_grad=True by default
        model.aux_classifier._modules['4'] = nn.Conv2d(256, 2, kernel_size=(1, 1), stride=(1, 1))
        model.classifier._modules['4'] = nn.Conv2d(256, 2, kernel_size=(1, 1), stride=(1, 1))

    # load trained weights to the model
    if cfg.hyperparameter.trained_weights:
        model.load_state_dict(torch.load(cfg.hyperparameter.trained_weights, map_location=device))

    # load the loss, optimizer and scheduler
    criterion = train_utils.loss_functions()
    optimizer = torch.optim.Adam(model.parameters(), cfg.hyperparameter.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    metrics = train_utils.Metrics(cfg.hyperparameter.num_classes)

    # arrays to store and plot loss and accuracy
    train_loss_global, train_accuracy_global, train_axis = [], [], []
    val_loss_global, val_accuracy_global, val_axis = [], [], []

    # print training info
    print(f'\n--- Date: {cfg.hyperparameter.date}\n'
          f'--- Type: {args.type}\n'
          f'--- Save this run: {cfg.hyperparameter.save_run}\n'
          f'--- Model: {cfg.hyperparameter.model}\n'
          f'--- Pretrained: {cfg.hyperparameter.pretrained}\n'
          f'--- Loaded weights: {cfg.hyperparameter.trained_weights}\n'
          f'--- Split ratio: {cfg.hyperparameter.split_ratio}\n'
          f'--- Patch level: {cfg.hyperparameter.patch_level}\n'
          f'--- Mask level: {cfg.hyperparameter.mask_level}\n'
          f'--- Patch size: {cfg.hyperparameter.patch_size}\n'
          f'--- Batch size: {cfg.hyperparameter.batch_size}\n'
          f'--- Learning rate: {cfg.hyperparameter.learning_rate}\n')

    # training process
    if args.type == 'train':
        print("Starting process of training...")
        process_of_training()
        print("Process of training is done!")

    # evaluating process
    if args.type == 'eval':
        print("Starting process of evaluation...")
        process_of_evaluation()
        print("Process of evaluation is done!")

    # learning rate searching process
    if args.type == 'find_lr':
        print("Starting learning rate searching process...")
        find_lr(train_loader)
        print("Process of searching learning rate is done!")

    # storing patch examples process
    if args.type == 'find_sample':
        print("Starting process of storing random patch samples...")
        find_random_samples()
        print("Process of storing random patch samples is done!")

    # storing extremal (with high and low loss) examples process
    if args.type == 'find_extreme':
        print("Starting process of storing extremal patch samples...")
        find_extremal_samples()
        print("Process of storing extremal patch samples is done!")
