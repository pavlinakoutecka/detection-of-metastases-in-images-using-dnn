from __future__ import division

"""
RESNET50
    - pretrained model
    - sorted to functions
    - using visualization
    - using random seed generator
    - with transforms
    - model was taken from https://pytorch.org/hub/pytorch_vision_resnet/
"""

# IMPORTS --------------------------------------------------------------------------------------------------------
import numpy as np
import pandas as pd
import os
import cv2
import random
import math

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import roc_curve, auc, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
from datetime import datetime
now = datetime.now()
import time
from PIL import Image

import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torch.autograd import Variable

import sys
import pathlib
parent_path = str(pathlib.Path().absolute().parent)
sys.path.append(parent_path)
import configuration as cfg
import train_utils
import plot_utils
import utils

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
# -----------------------------------------------------------------------------------------------------------------


# LOADING DATA  ---------------------------------------------------------------------------------------------------
train_labels_path = '/datagrid/Medical/microscopy/patch_camelyon/histopathologic_cancer_detection/train_labels.csv'
test_labels_path = '/datagrid/Medical/microscopy/patch_camelyon/histopathologic_cancer_detection/sample_submission.csv'
train_path = '/datagrid/Medical/microscopy/patch_camelyon/histopathologic_cancer_detection/train'
test_path = '/datagrid/Medical/microscopy/patch_camelyon/histopathologic_cancer_detection/test'


def load_data():
    train_labels = pd.read_csv(train_labels_path, error_bad_lines=False)
    test_labels = pd.read_csv(test_labels_path, error_bad_lines=False)

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


# TRAINING & TESTING & CHECKING ACCURACY ----------------------------------------------------------------------------

# train the model
def train_epoch(train_loader, model, criterion, optimizer, epoch, train_loss, train_acc, train_axis):

    # object to store & plot the losses
    losses = train_utils.Average()

    # compute estimated time till the end of the epoch
    startTime = time.time()

    # Train in mini-batches
    for batch_idx, (images, labels) in enumerate(train_loader):

        images = images.to(device)
        labels = labels.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward
        outputs = model(images)
        loss = criterion(outputs, labels)
        _, predicted = torch.max(outputs, 1)

        # backward + optimize
        losses.update(loss.data.cpu().numpy(), labels.size(0))
        loss.backward()
        optimizer.step()

        # compute estimated time till the end of epoch
        if batch_idx % 10 == 0:
            stopTime = time.time()
            rem_seconds = (stopTime - startTime) * (len(train_loader) - batch_idx) / 10
            rem_minutes, rem_seconds = divmod(rem_seconds, 60)
            startTime = time.time()

        # save predicted to compute accuracy
        if batch_idx == 0:
            out = predicted.data.cpu().numpy()
            label = labels.cpu().numpy()
        else:
            out = np.concatenate((out, predicted.data.cpu().numpy()),axis=0)
            label = np.concatenate((label, labels.cpu().numpy()),axis=0)

        # plot loss and accuracy after bunch of mini-batches have finished
        if batch_idx % 4 == 0 and batch_idx != 0:
            acc = np.sum(out == label) / len(out)
            train_acc.append(acc)
            train_loss.append(losses.avg)
            train_axis.append(epoch+batch_idx/len(train_loader))

        # print info
        print('EPOCH {} [{}/{} ({:.0f}%)]\t'
              '\tActual Loss: {:.4f} \t({:.4f})\t'
              'Estimated time till the end of epoch: {:02.0f}:{:02.0f} min\t'.format(
            epoch, batch_idx, len(train_loader), 100. * batch_idx / len(train_loader), losses.val, losses.avg, rem_minutes, rem_seconds), end="\r", flush=True)

    # accuracy
    acc = np.sum(out == label) / len(out)

    # return acc and loss as the train outcome
    return acc, losses.avg


# test and evaluate the model
def val_epoch(val_loader, model, criterion, epoch, val_loss, val_acc, val_axis):

    # object to store & plot the losses
    losses = train_utils.Average()

    # switch to evaluation mode (turning off gradients to speed up)
    model.eval()
    with torch.no_grad():

        # mini-batches
        for batch_idx, data in enumerate(val_loader):

            images, labels = data
            images = images.to(device)
            labels = labels.to(device)

            # forward + backward + optimize
            outputs = model(images)
            probability = torch.nn.functional.softmax(outputs, dim=1)  # recompute probabilities from the outputs
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs, 1)

            losses.update(loss.data.cpu().numpy(), labels.size(0))

            # save predicted to compute accuracy
            if batch_idx == 0:
                out = predicted.data.cpu().numpy()
                label = labels.cpu().numpy()
                prob = probability[:, 1].cpu().numpy()
            else:
                out = np.concatenate((out, predicted.data.cpu().numpy()), axis=0)  # predicted
                label = np.concatenate((label, labels.cpu().numpy()), axis=0)  # true
                prob = np.concatenate((prob, probability[:, 1].cpu().numpy()), axis=0)  # predicted probability of '1'

            # plot loss and accuracy after bunch of mini-batches have finished
            if batch_idx % 4 and batch_idx != 0:
                acc = np.sum(out == label) / len(out)
                val_acc.append(acc)
                val_loss.append(losses.avg)
                val_axis.append(epoch + batch_idx / len(val_loader))

        # evaluate ROC
        fpr, tpr, _ = roc_curve(label, prob)  # input: real labels, probability of '1'
        roc_auc = auc(fpr, tpr)

        # evaluate accuracy
        acc = np.sum(out == label)/len(out)

        # evaluate confusion matrix
        cm = confusion_matrix(label, out)

        # return acc and loss as the validation outcome
        return acc, losses.avg, cm, roc_auc, fpr, tpr


# run the train process of training, testing and evaluating the model
def train_process(epochs, train_loader, val_loader):

    # values to store best train and validation accuracy
    best_acc_train = float(0)
    best_acc_val = float(0)

    # arrays to store & plot the loss and accuracy
    train_loss, train_acc, train_axis = [],[],[]
    val_loss, val_acc, val_axis = [],[],[]

    for epoch in range(epochs):

        # compute a training epoch
        acctrain, losstrain = train_epoch(train_loader, model, criterion, optimizer, epoch, train_loss, train_acc, train_axis)

        # compute a validation epoch
        accval, lossval, cm, roc_auc, fpr, tpr = val_epoch(val_loader, model, criterion, epoch, val_loss, val_acc, val_axis)

        # reduce LR when a metric (loss) has stopped decreasing
        scheduler.step(lossval)

        # print training and validation info
        best_acc_train = max(acctrain, best_acc_train)
        best_acc_val = max(accval, best_acc_val)
        print('\n\t>>> TRAIN\t accuracy: %f (best) - %f (current)\t  loss: %f' % (best_acc_train, acctrain, losstrain))
        print('\t>>> VALIDATION\t accuracy: %f (best) - %f (current)\t  loss: %f\n' % (best_acc_val, accval, lossval))

        # save models checkpoint
        torch.save(model.state_dict(), "./trained_models/" + now.strftime("%Y%d%m-%H%M%S__") + model_name + "__ep__" + str(epoch) + ".pt")

        # plot and save loss curve
        df = pd.DataFrame({'axis':train_axis,'loss':train_loss})
        df.to_csv('train_loss' + str(epoch) + '.csv', header=False, index=False)
        df = pd.DataFrame({'axis':val_axis,'loss':val_loss})
        df.to_csv('val_loss' + str(epoch) + '.csv', header=False, index=False)
        plt.figure()
        plt.plot(train_axis,train_loss, "b-", label='training loss')
        plt.plot(val_axis, val_loss, "r-", label='validation loss')
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.title("Model loss")
        plt.legend()
        plt.savefig('./graphs/' + 'model_loss' + str(epoch) + '.png')
        plt.show()

        # plot and save accuracy curve
        df = pd.DataFrame({'axis':train_axis,'acc':train_acc})
        df.to_csv('train_acc' + str(epoch) + '.csv', header=False, index=False)
        df = pd.DataFrame({'axis':val_axis,'acc':val_acc})
        df.to_csv('val_acc' + str(epoch) + '.csv', header=False, index=False)
        plt.figure()
        plt.plot(train_axis,train_acc, "b-", label='training accuracy')
        plt.plot(val_axis, val_acc, "r-", label='validation accuracy')
        plt.xlabel("epoch")
        plt.ylabel("accuracy")
        plt.title("Model accuracy")
        plt.legend()
        plt.savefig('./graphs/' + 'model_accuracy' + str(epoch) + '.png')
        plt.show()

        # plot confusion matrix
        plot_utils.plot_confusion_matrix(epoch, cm=cm, normalize=False, title="Confusion matrix", cmap=None, target_names=['0', '1'])

        # plot ROC curve
        plt.figure()
        plt.plot(fpr, tpr, color='red', lw=2, label='ROC curve (area = %0.4f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC curve')
        plt.legend(loc="lower right")
        plt.savefig('./graphs/' + 'roc_curve' + str(epoch) + '.png')
        plt.show()
# ------------------------------------------------------------------------------------------------


# FINDING THE LEARNING RATE ----------------------------------------------------------------------
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
        outputs = model(inputs)
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
    plt.figure(figsize=(6.5, 4))
    plt.plot(log_lrs[10:-5], losses[10:-5], linewidth=1.5, color='crimson')
    plt.xlabel("Learning rate")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.title(f"Learning rate searching process for the {cfg.hyperparameter.model_title} model", y=1.02)
    plt.savefig(cfg.path.graphs + f'lr_searcher' + cfg.hyperparameter.saving_string + '.pdf', format='pdf')
    plt.show()
# ------------------------------------------------------------------------------------------------


if __name__ == '__main__':

    # load data and device
    train, val = load_data()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # '.to(device)' can be written also as '.cuda()'

    # load the model
    model_name = 'resnet50trans'
    model = torch.hub.load('pytorch/vision', 'resnet50', pretrained=True)
    # model.weights_initialization() # NOT FOR PRETRAINED MODELS
    path = '/mnt/medical/microscopy/patch_camelyon/nets_bakproj/trained_models/20193112-095359__resnet50trans___FINAL_MODEL/20193112-095359__resnet50trans__ep__33.pt'  # loading my own model
    model.load_state_dict(torch.load(path, map_location=device))  # loading my own model
    model.to(device)

    # hyperparameters for the model
    epochs = 40
    classes = 2
    batch_size = 64
    learning_rate = 0.0003
    training = False  # 'True' if training the model should be happening
    finding_lr = True  # 'True' if finding the optimal learning rate should be happening

    # data transforms
    train_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(224),
        # transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=1),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # load dataset in required format
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

    # training process
    if training:
        print("Starting process of training...")
        train_process(epochs, loader_train, loader_val)
        print("Process of training is done!")

    # learning rate searching process
    if finding_lr:
        print("Starting learning rate searching process...")
        find_lr(loader_train)
        print("Process of searching learning rate is done!")

