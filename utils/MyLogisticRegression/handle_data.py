import numpy as np
from .stats import *
import copy
import math

def NAN_to_median(values):
    no_nan = values[~np.isnan(values)]
    median_ = median(no_nan)
    i = 0
    l = 0
    values[np.isnan(values)] = median_
    return values

def add_polynomial_features(x, power):
    init = x
    for pow in range(1, power + 1):
        x = np.column_stack((x, init ** pow))
    return x


def split_x_y(set):
    set = set.transpose()
    set_x = set[:-1]
    set_y = set[-1:]
    return (set_x.transpose(), set_y.transpose())


def data_spliter(x, y, proportion=0.8):
    shuffle = np.column_stack((x, y))
    np.random.shuffle(shuffle)
    training_lenght = int(shuffle.shape[0] // (1/proportion))
    if training_lenght == 0:
        training_lenght = 1
    training_set = shuffle[:training_lenght]
    test_set = shuffle[training_lenght:]
    return split_x_y(training_set) + split_x_y(test_set) #x_train = data[0], y_train = data[1], x_test = data[2], y_test = data[3]


def minmax_normalization(x_values):
    i = 0
    x_values = x_values.transpose()
    while i < x_values.shape[0]:
        max = float(np.nanmax(x_values[i]))
        min = float(np.nanmin(x_values[i]))
        range = max - min
        x_values[i] = np.divide(np.subtract(x_values[i], min), range)
        i += 1
    return x_values.transpose()

#The correct class needs to be represented numerically with 1 or "yes", while the wrong answer needs to represented as 0
#When doing multiclassification, you need to let each class clasify against all the others
#and find the one with the highest probablity,
#to do that you need to set the class you want as 1 or "yes" and all other classes as 0 or "no"
def descriminate_classes(expected_values, class_):
    i = 0
    y = copy.deepcopy(expected_values)
    while i < expected_values.shape[0]:
        if y[i] == class_:
            y[i] = 1
        else:
            y[i] = 0
        i += 1
    return y

def what_category(data, categories):
    for i, category in enumerate(categories, 1):
        if data == category:
            return i
    print("ERROR: Missing category in categorical_data_to_numerical_data: " + str(data))
    exit()

#Creates bug in nan_to_median function for unknown reason
def categorical_data_to_numerical_data(categories, column):
    for i, data in enumerate(column):
        column[i] = what_category(data, categories)
    return column

