import numpy as np
import pandas as pd
import sys
from utils.logistic_regression import *
from utils.logistic_metrics import *
from utils.handle_data import *
import copy
from utils.test import *


def main():
    houses = np.array(["Ravenclaw", "Slytherin", "Gryffindor","Hufflepuff"])
    thetas = []
    x_values = pd.read_csv("datasets/" + sys.argv[1], index_col=0)
    x_values = np.array(x_values.drop(["Hogwarts House", "First Name", "Last Name", "Birthday", "Best Hand", "Arithmancy", "Care of Magical Creatures", "Astronomy"], axis=1))
    x_values = NAN_to_median(x_values)
    x_values = minmax_normalization(x_values)
    x_values = add_polynomial_features(x_values, 2)
    expected_values = np.array(pd.read_csv("datasets/" + sys.argv[1])[["Hogwarts House"]])
    data = data_spliter(x_values, expected_values)
    for house in houses:
        y = descriminate_classes(data[1], house)
        LR = LogisticRegression(type="mini_batch", thetas=np.zeros((x_values.shape[1] + 1, 1)), alpha=0.1, n_cycle=2000000, lambda_=0.1)
        print(house)
        binary_feedback(LR, data[0], y)
        LR.fit_(data[0], y)
        binary_feedback(LR, data[0], y)
        thetas = assemble_thetas(thetas, LR.theta)
    pd.DataFrame(thetas, columns=houses).to_csv("datasets/theta.csv")
    final_feedback(data[2], data[3])


if __name__ == "__main__":
    main()
