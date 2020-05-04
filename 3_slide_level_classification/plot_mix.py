"""
FOR PERSONAL PURPOSES
"""


import numpy as np
import matplotlib.pyplot as plt
import csv
import matplotlib.image as mpimg
import matplotlib.cm as mpcm
from matplotlib.colors import NoNorm

import sys
import pathlib
parent_path = str(pathlib.Path().absolute().parent)
sys.path.append(parent_path)
import configuration as cfg
from laplotter import LossAccPlotter


def plot_smoothed_curve():

    # lvl1
    # ep3__val_loss__ps256__sr1__09_04-13_37_27.csv
    # ep0__val_loss__ps256__sr1__10_04-11_35_35.csv
    #
    #
    # lvl2
    # ep3__val_loss__ps256__sr1__10_04-11_38_57.csv
    # ep2__train_accuracy__ps256__sr1__11_04-11_41_49.csv

    path = cfg.path.graphs + 'smoothed' + cfg.hyperparameter.saving_string + '.png'
    train_acc_csv = cfg.path.graphs + 'ep3__train_accuracy__ps256__sr1__10_04-11_38_57.csv'
    val_acc_csv = cfg.path.graphs + 'ep3__val_accuracy__ps256__sr1__10_04-11_38_57.csv'
    train_loss_csv = cfg.path.graphs + 'ep3__train_loss__ps256__sr1__10_04-11_38_57.csv'
    val_loss_csv = cfg.path.graphs + 'ep3__val_loss__ps256__sr1__10_04-11_38_57.csv'
    train_acc_csv2 = cfg.path.graphs + 'ep2__train_accuracy__ps256__sr1__11_04-11_41_49.csv'
    val_acc_csv2 = cfg.path.graphs + 'ep2__val_accuracy__ps256__sr1__11_04-11_41_49.csv'
    train_loss_csv2 = cfg.path.graphs + 'ep2__train_loss__ps256__sr1__11_04-11_41_49.csv'
    val_loss_csv2 = cfg.path.graphs + 'ep2__val_loss__ps256__sr1__11_04-11_41_49.csv'

    plotter = LossAccPlotter(title="Model performance",
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

    train_acc2 = np.genfromtxt(train_acc_csv2, delimiter=',')
    val_acc2 = np.genfromtxt(val_acc_csv2, delimiter=',')
    train_loss2 = np.genfromtxt(train_loss_csv2, delimiter=',')
    val_loss2 = np.genfromtxt(val_loss_csv2, delimiter=',')

    for index, _ in enumerate(train_acc2):
        plotter.add_values(train_acc2[index, 0] + 4, acc_train=train_acc2[index, 1], loss_train=train_loss2[index, 1], redraw=False)
    for index, _ in enumerate(val_acc2):
        plotter.add_values(val_acc2[index, 0] + 4, acc_val=val_acc2[index, 1], loss_val=val_loss2[index, 1], redraw=False)

    # redraw once at the end
    plotter.redraw()
    plotter.block()


def plot_patch_prediction():
    """

    :return:
    """

    path = 'C:/Users/koute/ownCloud/BAKALÁŘKA/scripts/bakprac/3_slide_level_classification/extreme_losses/'

    for i in range(100):

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))

        patch = mpimg.imread(path + 'lowest_loss_patch' + str(i) + '.png')
        mask = mpimg.imread(path + 'lowest_loss_mask' + str(i) + '.png')
        prediction = mpimg.imread(path + 'lowest_loss_prediction' + str(i) + '.png')

        ax1.imshow(patch)
        ax2.imshow(mask, cmap='gray', norm=NoNorm())
        ax3.imshow(prediction, cmap='gray', norm=NoNorm())

        ax1.set_title('Patch')
        ax2.set_title('True mask')
        ax3.set_title('Predicted mask')

        fig.show()
        fig.savefig('C:/Users/koute/ownCloud/BAKALÁŘKA/scripts/bakprac/3_slide_level_classification/extreme_losses/merged_lowest_loss/' + str(i) + '.png')


def plotFROC():
    """Plots the FROC curve

    Args:
        total_FPs:      A list containing the average number of false positives
        per image for different thresholds

        total_sensitivity:  A list containig overall sensitivity of the system
        for different thresholds

    Returns:
        -
    """

    csvDIR1 = cfg.hyperparameter.saving_folder + 'FROC_data_lvl_1.csv'
    csv1 = np.genfromtxt(csvDIR1, delimiter=',')
    csvDIR2 = cfg.hyperparameter.saving_folder + f'FROC_data_lvl_2.csv'
    csv2 = np.genfromtxt(csvDIR2, delimiter=',')

    total_FPs1, total_sensitivity1, FROC_score1 = csv1.T[0], csv1.T[1], csv1.T[2]
    total_FPs2, total_sensitivity2, FROC_score2 = csv2.T[0], csv2.T[1], csv2.T[2]

    x_limit = 8  # min(total_FPs1[1], total_FPs2[1])

    fig = plt.figure()
    plt.xlabel('Average number of False Positives', fontsize=12)
    plt.ylabel('Metastasis detection sensitivity', fontsize=12)
    fig.suptitle('Free response receiver operating characteristic curve', fontsize=12)
    plt.plot(total_FPs1, total_sensitivity1, '-', color='red', label='level 1 FROC curve (Final score = {:.4f}'.format(FROC_score1[1]))
    plt.plot(total_FPs2, total_sensitivity2, '-', color='blue', label='level 2 FROC curve (Final score = {:.4f}'.format(FROC_score2[1]))
    plt.xlim(right=x_limit)
    plt.legend(loc='lower right')
    plt.grid(b=True, which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.savefig(cfg.path.images + 'Evaluation_FROC.png')
    plt.show()

