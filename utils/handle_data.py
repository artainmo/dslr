import numpy as np
import utils.stat as stat

def NAN_to_median(values):
    no_nan = values[~np.isnan(values)]
    median = stat.median_(no_nan)
    i = 0
    l = 0
    values[np.isnan(values)] = median
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
    return split_x_y(training_set) + split_x_y(test_set)


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
