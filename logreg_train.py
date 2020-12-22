import numpy as np
import pandas as pd
import sys
from utils.MyLinearRegression import *

def get_xy_values(path, classes_label):
    try:
    	x_values = pd.read_csv(path, index_col=0)
    except:
        print("Error: argument file")
        exit()
    x_values = np.array(x_values.drop(["Hogwarts House", "First Name", "Last Name", "Birthday", "Best Hand", "Arithmancy", "Care of Magical Creatures", "Astronomy"], axis=1)) #non-numerical data is dropped too, even if some could be converted to numerical, by assigning key values to them
    x_values = NAN_to_median(x_values)
    x_values = minmax_normalization(x_values)
    x_values = add_polynomial_features(x_values, 4)
    expected_values = np.array(pd.read_csv(path)[[classes_label]])
    return x_values, expected_values

def assemble_thetas(thetas, theta):
    try:
        thetas = np.column_stack((thetas, theta))
    except:
        thetas = theta
    return thetas

if __name__ == "__main__":
    houses = np.array(["Ravenclaw", "Slytherin", "Gryffindor","Hufflepuff"])
    thetas = []
    if len(sys.argv) != 2:
        print("Error: number arguments")
        exit()
    x_values, expected_values = get_xy_values(sys.argv[1], "Hogwarts House")
    data = data_spliter(x_values, expected_values, 1)
    for house in houses:
        y = descriminate_classes(data[1], house)
        LR = LogisticRegression(type="mini_batch", thetas=np.zeros((x_values.shape[1] + 1, 1)), alpha=1, n_cycle=200015, lambda_=0.1)
        print('\033[96m' + house + '\033[0m')
        feedback(LR, data[0], y)
        LR.fit(data[0], y)
        feedback(LR, data[0], y)
        thetas = assemble_thetas(thetas, LR.theta)
    pd.DataFrame(thetas, columns=houses).to_csv("datasets/theta.csv")
    print('\033[96m' + "Final test with train set"+ '\033[0m')
    sklearn_feedback(class_answers(houses, data[0], pd.read_csv("datasets/theta.csv")), data[1])
    if data[2].shape[0] != 0:
        print('\033[96m' + "Final test with test set"+ '\033[0m')
        sklearn_feedback(class_answers(houses, data[2], pd.read_csv("datasets/theta.csv")), data[3])

#0.9 split - 4poly - 1alpha - ncycle200015 - lambda0.1 - 3min -> ||/ -> Each miss has high test set but not enough train -> use all data for train
#1 split - 4poly - 1alpha - ncycle200015 - lambda0.1 -> result 0.98125
