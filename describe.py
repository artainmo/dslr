import sys
import pandas as pd
import numpy as np
from utils.stat import *


def iterate_data(data, func):
    ret = np.array([])
    for column in data:
        ret = np.append(ret, func(column))
    return ret


def get_numerical_data():
    i = 0
    column_labels = []
    try:
    	data = pd.read_csv("datasets/" + sys.argv[1], index_col=0)
    except:
        print("Error with file argument")
        exit()
    for (column_name, column_data) in data.iteritems():
        if isinstance(column_data[0], (int, float)) == True:
            try:
                numerical_data = np.append(numerical_data, np.array([column_data]), axis=0)
            except:
                numerical_data = np.array([column_data])
            column_labels.append(column_name)
    return (numerical_data, column_labels)


if __name__=="__main__":
    row_labels = ["count", "min", "max", "mean", "standard_derivation", "quartiles_25", "median", "quartiles_75", "mode", "skewness", "kurtosis"]
    functions = [count_, min_, max_, mean_, standard_derivation, quartiles_25, median_, quartiles_75, mode_, skewness_, kurtosis_]
    numerical_data, column_labels = get_numerical_data()
    for func in functions:
        try:
            rows = np.append(rows, np.array([iterate_data(numerical_data, func)]), axis=0)
        except:
            rows = np.array([iterate_data(numerical_data, func)])
    print(pd.DataFrame(rows, index=row_labels, columns=column_labels))
