from __future__ import division


# IMPORTS --------------------------------------------------------------------------------------------------------
import numpy as np
import pandas as pd
import os
import cv2
import random
import math

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
from datetime import datetime
now = datetime.now()
import time
import utils

import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torch.autograd import Variable

from torchvision.utils import save_image
import torch
import torchvision

# random seed initialization
manualSeed = 1
np.random.seed(manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
# if you are using GPU
torch.cuda.manual_seed(manualSeed)
torch.cuda.manual_seed_all(manualSeed)
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
# ------------------------------------------------------------------------------------------------------------------

# LOADING DATA  ---------------------------------------------------------------------------------------------------
train_labels_path = '/datagrid/Medical/microscopy/patch_camelyon/histopathologic_cancer_detection/train_labels.csv'
test_labels_path = '/datagrid/Medical/microscopy/patch_camelyon/histopathologic_cancer_detection/sample_submission.csv'
train_path = '/datagrid/Medical/microscopy/patch_camelyon/histopathologic_cancer_detection/train'
test_path = '/datagrid/Medical/microscopy/patch_camelyon/histopathologic_cancer_detection/test'

def load_data():
    train_labels = pd.read_csv(train_labels_path, error_bad_lines=False)

    print(f'\nThere are {len(os.listdir(train_path))} pictures in train.')  # prints formatted string literals
    print(f'There are {len(os.listdir(test_path))} pictures in test.')

    # splitting data into train and val
    train, val = train_test_split(train_labels, stratify=train_labels.label, test_size=0.1, shuffle=True)
    # one sample for example: 214338 b20ceff0cc4d20ceart56 1
    # stratify - parameter, makes a split so that the proportion of values in the sample produced will be the same
    #            as the proportion of values provided to this parameter

    print(f'\nAfter split: there are {len(train)} pictures in train.')  # prints formatted string literals
    print(f'After split: there are {len(val)} pictures in val.')

    # checking for disbalance
    print('\nChecking for disbalance...')
    print('0:', train_labels.label.value_counts()[0], ', 1:', train_labels.label.value_counts()[1])
    print('No disbalance here!')

    return train, val
# ------------------------------------------------------------------------------------------------------------------


# PREPARING DATASET CLASS ------------------------------------------------------------------------------------------
class MyDataset(Dataset):
    def __init__(self, data, data_dir='./', transform=None):
        # data: train or val (contains image name and its label) - splitted before
        super().__init__()
        self.data = data.values  # from train or val take img_name and img_label
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_name, img_label = self.data[index]
        img_path = os.path.join(self.data_dir, img_name + '.tif')
        img = cv2.imread(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return img, img_label
# ------------------------------------------------------------------------------------------------------------------

def img_denorm(img, mean, std):
    # for ImageNet the mean and std are:
    # mean = np.asarray([ 0.485, 0.456, 0.406 ])
    # std = np.asarray([ 0.229, 0.224, 0.225 ])

    denormalize = transforms.Normalize((-1 * mean / std), (1.0 / std))

    res = img.squeeze(0)
    res = denormalize(res)

    # Image needs to be clipped since the denormalize function will map some
    # values below 0 and above 1
    res = torch.clamp(res, 0, 1)
    return (res)


if __name__ == '__main__':

    # load data and device
    train, val = load_data()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # '.to(device)' can be written also as '.cuda()'

    # load the model
    model_name = 'resnet50trans'
    model = torch.hub.load('pytorch/vision', 'resnet50', pretrained=True)
    # model.weights_initialization() # NOT FOR PRETRAINED MODELS
    path = os.path.dirname(
        os.path.realpath(__file__)) + '/trained_models/20193112-095359__resnet50trans___FINAL_MODEL/20193112-095359__resnet50trans__ep__33.pt'  # loading my own model
    model.load_state_dict(torch.load(path, map_location=device))  # loading my own model
    model.to(device)

    # hyperparameters for the model TODO: try tuning these parameters
    epochs = 40
    classes = 2
    batch_size = 64
    learning_rate = 0.0003
    training = True  # 'True' if training the model should be happening
    finding_lr = False  # 'True' if finding the learning rate should be happening

    # data transforms TODO: try tuning these parameters
    train_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


    # load TEST dataset in required format
    test_labels = pd.read_csv(test_labels_path, error_bad_lines=False)
    dataset_test = MyDataset(data=test_labels, data_dir=test_path, transform=val_transforms)
    loader_test = DataLoader(dataset=dataset_test, batch_size=batch_size // 2, shuffle=False, num_workers=16)

    # load TRAIN and VAL dataset in required format
    dataset_train = MyDataset(data=train, data_dir=train_path, transform=train_transforms)
    dataset_val = MyDataset(data=val, data_dir=train_path, transform=val_transforms)

    # cut data to learn faster (just for exploration)
    # use ratio = 1 for full dataset
    ratio = 1
    train_split_len = len(dataset_train) // ratio
    val_split_len = len(dataset_val) // ratio
    part_train = torch.utils.data.random_split(dataset_train, [train_split_len, len(dataset_train) - train_split_len])[0]
    part_val = torch.utils.data.random_split(dataset_val, [val_split_len, len(dataset_val) - val_split_len])[0]

    loader_train = DataLoader(dataset=part_train, batch_size=batch_size, shuffle=True, num_workers=16)
    loader_val = DataLoader(dataset=part_val, batch_size=batch_size // 2, shuffle=True, num_workers=16)

    print(f'\nAfter cut: there are {len(part_train)} pictures in train.')  # prints formatted string literals
    print(f'After cut: there are {len(part_val)} pictures in val.')

    # loss, optimizer and scheduler initialization
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    # SUBMISSION
    model.eval()
    with torch.no_grad():

        # mini-batches
        for batch_idx, data in enumerate(loader_test):

            # print info
            print('[{}/{} ({:.0f}%)]'.format(batch_idx, len(loader_test), 100. * batch_idx / len(loader_test), end="\r", flush=True))

            images, labels = data
            images = images.to(device)
            labels = labels.to(device)

            # forward + backward + optimize
            outputs = model(images)
            probability = torch.nn.functional.softmax(outputs, dim=1)  # recompute probabilities from the outputs
            _, predicted = torch.max(outputs, 1)

            # # plot most correct and incorrect classes
            # pred = predicted.data.cpu().numpy()
            # prob = probability[:, 1].cpu().numpy()
            # label = labels.cpu().numpy()
            #
            # for i in range(len(images)):
            #
            #     if labels[i] != pred[i] and abs(label[i] - prob[i]) > 0.7:
            #         print(batch_idx, i, prob[i], pred[i],label[i])
            #
            #         im = images[i]
            #         im_denorm = img_denorm(im, np.asarray([0.485, 0.456, 0.406]), np.asarray([0.229, 0.224, 0.225]))
            #         save_image(im_denorm, 'bad' + str(batch_idx) + '_' + str(i)+'.png')


            # save predicted to compute accuracy
            if batch_idx == 0:
                out = predicted.data.cpu().numpy()
                label = labels.cpu().numpy()
                prob = probability[:, 1].cpu().numpy()
            else:
                out = np.concatenate((out, predicted.data.cpu().numpy()), axis=0)  # predicted
                label = np.concatenate((label, labels.cpu().numpy()), axis=0)  # true
                prob = np.concatenate((prob, probability[:, 1].cpu().numpy()), axis=0)  # predicted probability of '1'


    # save labels and ids
    sample_df = pd.read_csv(test_labels_path)
    sample_list = list(sample_df.id)
    df_sub = pd.DataFrame({'id': sample_list, 'label': prob})

    # export to csv
    df_sub.to_csv('test_labels.csv', header=True, index=False)

