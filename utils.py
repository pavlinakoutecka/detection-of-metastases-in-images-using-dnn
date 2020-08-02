"""
File:           utils.py
Author:         Pavlina Koutecka
Date:           25/02/2020
Description:    This file

"""


import sys
import time
import os
from os import path
import argparse
import numpy as np

last_time = time.time()
begin_time = last_time


def format_time(seconds):
    """
    This function converts given number of seconds into nice format.

    :param seconds: number of seconds to be formatted
    :return: formatted time

    Inspired by https://github.com/JooHyun-Lee/Camelyon17/blob/master/utils.py.
    """

    seconds = seconds
    hours = int(seconds / 3600)
    seconds = seconds - hours * 3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes * 60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds * 1000)

    f = ''
    i = 1

    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


def visualize_progress(current, total, estimated=None, text=None):
    """
    This function prints progress bar of some running process to standard output.

    :param current: current number
    :param total: total number
    :param estimated: estimated time till the end of process
    :param text: text to be printed with progress bar
    :return:

    Inspired by https://github.com/JooHyun-Lee/Camelyon17/blob/master/utils.py.
    """

    global last_time, begin_time
    if current == 0:
        begin_time = time.time()

    # bar's parameters
    total_len = 60.
    _, term_width = os.popen('stty size', 'r').read().split()
    term_width = int(term_width)

    cur_len = int(total_len * current / total)
    rest_len = int(total_len - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    total_time = cur_time - begin_time

    L = []
    if text:
        L.append(str(text))
    L.append(' | Step: %s' % format_time(step_time))
    L.append(' | Total: %s' % format_time(total_time))
    if estimated:
        L.append(' | ETA: %s' % format_time(estimated))
    else:
        L.append(' | ETA: Unknown')

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width - int(total_len) - len(msg) - 3):
        sys.stdout.write(' ')

    # go back to the center of the bar
    for i in range(term_width - int(total_len / 2) + 2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current + 1, total))

    if current < total - 1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')

    sys.stdout.flush()


def rename(folder):
    """

    :param folder:
    :return:
    """

    os.chdir(folder)

    for filename in os.listdir(folder):
        os.rename(filename, filename.replace('.tif', ''))

    # create list of available slides
    files = [f for f in os.listdir(folder) if path.isfile(path.join(folder, f))]
    print(files[0])


def find_nearest(array, value):
    """

    :param array:
    :param value:
    :return:
    """

    array = np.asarray(array)
    index = (np.abs(array - value)).argmin()

    return index


def parse_args():
    """
    This function parses passing arguments.

    :return: parsed arguments
    """

    # creating an ArgumentParser object
    parser = argparse.ArgumentParser()

    # process passing arguments and return them
    parser.add_argument('-t', '--type')
    parser.add_argument('-s', '--start')
    parser.add_argument('-e', '--end')
    parser.add_argument('-d', '--data')
    return parser.parse_args()
