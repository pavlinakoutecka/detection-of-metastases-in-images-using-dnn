"""
File:           train_utils.py
Author:         Pavlina Koutecka
Date:           28/03/2020
Description:    This file

"""


import numpy as np
import cv2
import sklearn.metrics
from datetime import datetime
now = datetime.now()

import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional

import sys
import pathlib
parent_path = str(pathlib.Path().absolute().parent)
sys.path.append(parent_path)
import configuration as cfg


class Average(object):
    """
    This class computes and stores the average and current value.

    """
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class TrainDataset(Dataset):
    """
    This class extract patches from training dataset.

    """

    def __init__(self, data, transform=None, normalize=None):
        super().__init__()
        self.data = data.values  # take patch and its mask
        self.transform = transform
        self.normalize = normalize

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):

        # load the .csv values
        patch, mask = self.data[index]

        # load patch and its mask
        patch_img = cv2.imread(patch)
        mask_img = cv2.imread(mask, cv2.IMREAD_GRAYSCALE)

        # apply data transformations
        if self.transform:
            transforms = self.transform(image=patch_img, mask=mask_img)
            patch_img, mask_img = transforms['image'], transforms['mask']

        # normalize patches (required by the DeepLab architecture)
        if self.normalize:
            patch_img = self.normalize(patch_img)

        # return patch and its mask
        return patch_img, mask_img.squeeze(1).long()


class EvalDataset(Dataset):
    """
    This class extract patches from evaluating dataset.

    """

    def __init__(self, data, transforms=None):
        super().__init__()
        self.data = data  # take the patch
        self.transform = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        # load the .csv values
        patch_img = self.data[idx]

        # apply data transformations
        if self.transform:
            transforms = self.transform(image=patch_img)
            patch_img = transforms['image']

        # return patch
        return patch_img


class Metrics:
    """
    This class returns metrics specified for our task. It can initialize the metrics, compute it,
    update it, print it and reset it. Currently are implemented these return values:

        > accuracy
        > sensitivity, specificity, precision, recall
        > IOU, dice coefficient
        > confusion matrix

    """

    def __init__(self, num_classes):

        self.num_classes = num_classes

        self.TP = np.zeros(1, dtype='u8')  # true positive
        self.TN = np.zeros(1, dtype='u8')  # true negative
        self.FP = np.zeros(1, dtype='u8')  # false positive
        self.FN = np.zeros(1, dtype='u8')  # false negative
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), 'u8')

    def update(self, predictions, masks):

        target = masks.cpu().numpy()
        input = predictions.cpu().numpy()
        input = np.argmax(input, 1)  # first dimension are probabilities/scores
        _, predicted = torch.max(predictions.data, 1)

        confusion_matrix = sklearn.metrics.confusion_matrix(target.flatten(), input.flatten())

        TN = confusion_matrix[0][0]
        FN = confusion_matrix[1][0]
        TP = confusion_matrix[1][1]
        FP = confusion_matrix[0][1]

        np.add(self.confusion_matrix, confusion_matrix, out=self.confusion_matrix, casting="unsafe")
        self.TP += TP
        self.FP += FP
        self.FN += FN
        self.TN += TN

    def compute(self):

        TP, TN, FP, FN = self.TP, self.TN, self.FP, self.FN
        confusion_matrix = self.confusion_matrix

        with np.errstate(all='ignore'):  # in the case of dividing by zero - ignore these cases

            accuracy = float((TP+TN)/(TP+TN+FP+FN))
            sensitivity = TP / (TP + FN)
            specificity = TN / (TN + FP)
            precision = TP / (TP + FP)
            recall = TP / (TP + FN)
            iou = TP / (TP + FP + FN)
            dice_coefficient = 2. * precision * recall / (precision + recall)

        return accuracy, sensitivity, specificity, precision, recall, iou, dice_coefficient, confusion_matrix

    def print(self):

        accuracy, sensitivity, specificity, precision, recall, iou, dice_coefficient, confusion_matrix = self.compute()

        print("Confusion matrix:\n", confusion_matrix)
        for item in range(self.num_classes):
            print(f"Class: {str(cfg.hyperparameter.classes[item]):10s}\t"
                  f"Sensitivity: {sensitivity[item]:.5f}\t"
                  f"Specificity: {specificity[item]:.5f}\t"
                  f"Precision: {precision[item]:.5f}\t"
                  f"Recall: {recall[item]:.5f} \t"
                  f"IOU: {iou[item]:.5f}\t"
                  f"Dice coefficient: {dice_coefficient[item]:.5f}")

    def reset(self):

        self.TP = np.zeros(1, dtype='u8')  # true positive
        self.TN = np.zeros(1, dtype='u8')  # true negative
        self.FP = np.zeros(1, dtype='u8')  # false positive
        self.FN = np.zeros(1, dtype='u8')  # false negative
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), 'u8')


def loss_functions():
    """
    This
    Currenty are implemented these loss functions:

        > focal loss
        > cross-entropy loss
        > soft dice loss

    :return: chosen loss function
    """

    def focal_loss(prediction, masks):

        weights = np.asarray([0, 1])
        weights = torch.from_numpy(weights).to(prediction.device).float()
        prediction_prob = torch.nn.functional.softmax(prediction)

        # compute cross entropy
        ce = torch.nn.functional.cross_entropy(prediction, masks, reduction='none', weight=weights)[:, None, ...]

        # compute weight
        weight = (1 - prediction_prob.gather(1, masks[:, None, ...])) ** 2

        return torch.mean(weight * ce)

    def ce_loss(prediction, masks):

        weights = np.asarray([0, 1])
        weights = torch.from_numpy(weights).to(prediction.device).float()

        # cross entropy part
        result = torch.nn.functional.cross_entropy(prediction, masks, weight=weights)

        return result

    def soft_dice_loss(prediction, masks):

        smooth = 1.

        input = torch.nn.functional.softmax(prediction, dim=1)
        target = torch.nn.functional.one_hot(masks, num_classes=cfg.hyperparameter.num_classes).squeeze(1)
        target = target.permute(0, 3, 1, 2).contiguous().float()

        iflat = input.view(-1)
        tflat = target.view(-1)

        intersection = (iflat * tflat).sum()
        dice_coefficient = ((2. * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))

        return 1 - dice_coefficient

    return soft_dice_loss
