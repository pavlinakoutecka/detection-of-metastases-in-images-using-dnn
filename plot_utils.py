"""
File:           plot_utils.py
Author:         Pavlina Koutecka
Date:           09/04/2020
Description:    This file some basic utilities to improve the plotting process.
                Currently, these utilities are implemented:

                > plot confusion matrix - make a nice plot of a confusion matrix
                > plot smoothed curve - smoothen stored accuracy and loss curves.
"""


import matplotlib.pyplot as plt
import itertools
import numpy as np

import sys
import pathlib
parent_path = str(pathlib.Path().absolute().parent)
sys.path.append(parent_path)
import configuration as cfg
from laplotter import LossAccPlotter


def plot_confusion_matrix(cm, target_names, path, title='Confusion matrix', cmap=None, normalize=True):
    """
    Given a sklearn confusion matrix (cm), make a nice plot

    :param cm: confusion matrix from sklearn.metrics.confusion_matrix
    :param target_names: given classification classes such as [0, 1, 2]
                         the class names, for example: ['high', 'medium', 'low']
    :param title: the text to display at the top of the matrix
    :param cmap: the gradient of the values displayed from matplotlib.pyplot.cm
                 see http://matplotlib.org/examples/color/colormaps_reference.html
                 plt.get_cmap('jet') or plt.cm.Blues
    :param normalize:  If False, plot the raw numbers
                       If True, plot the proportions
    :return: confusion matrix as an object to plot

    References:
        http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
        https://www.kaggle.com/grfiv4/plot-a-confusion-matrix
    """

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(6, 4))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names)
        plt.yticks(tick_marks, target_names, rotation='vertical')

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylim([3.5, -0.5])  # [1.5, -0.5]
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(path)
    plt.show()


def plot_smoothed_curve(path, train_acc_csv, val_acc_csv, train_loss_csv, val_loss_csv):
    """
    Use the laplotter.py to smoothen some stored runs of accuracy and loss curves.

    :param path: path to where the result will be stored
    :param train_acc_csv: training accuracy curve to be smoothen
    :param val_acc_csv: validation accuracy curve to be smoothen
    :param train_loss_csv: training loss curve to be smoothen
    :param val_loss_csv: validation loss curve to be smoothen
    :return:
    """

    plotter = LossAccPlotter(title=f"Performance of the {cfg.hyperparameter.model_title}",
                             save_to_filepath=path,
                             show_regressions=False,
                             show_averages=True,
                             show_loss_plot=True,
                             show_acc_plot=True,
                             show_plot_window=True,
                             x_label="Epoch")

    train_acc = np.genfromtxt(train_acc_csv, delimiter=',')
    val_acc = np.genfromtxt(val_acc_csv, delimiter=',')
    train_loss = np.genfromtxt(train_loss_csv, delimiter=',')
    val_loss = np.genfromtxt(val_loss_csv, delimiter=',')

    for index, _ in enumerate(train_acc):
        plotter.add_values(train_acc[index, 0], acc_train=train_acc[index, 1], loss_train=train_loss[index, 1], redraw=False)
    for index, _ in enumerate(val_acc):
        plotter.add_values(val_acc[index, 0], acc_val=val_acc[index, 1], loss_val=val_loss[index, 1], redraw=False)

    # redraw once at the end
    plotter.redraw()
    plotter.block()
