import numpy as np
import pandas as pd
import sys
from utils.MyLogisticegression import *

def get_xy_values(path, classes_label):
    try:
    	x_values = pd.read_csv(path, index_col=0)
    except:
        print("Error: argument file")
        exit()
    x_values["Best Hand"].replace({"Right": 1, "Left": 2}, inplace=True)#transforming the "best hand" column from categorical to numerical data
    x_values = np.array(x_values.drop(["Hogwarts House", "First Name", "Last Name", "Birthday", "Arithmancy"], axis=1)) #We remove non-categorical textual data and Arithmancy (sole feature that does no affect the accuracy)
    x_values = NAN_to_median(x_values)
    x_values = minmax_normalization(x_values)
    x_values = add_polynomial_features(x_values, 0)
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
        LR = LogisticRegression(type="mini_batch", thetas=np.zeros((x_values.shape[1] + 1, 1)), alpha=1, n_cycle=5000, lambda_=0)
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

#1 split - 4poly - 1alpha - ncycle200015 - lambda0.1 - mini-batch18 - drop magical creatures, arithmancy and Astronomy -> train result 0.98125 -> test result 0.845
#same but without the drops -> train result 0.9768 -> test set result 0.955
#same but with 2poly and without the drops -> train result 0.97125 -> test result 0.938

#Same but with 2poly and "Best Hand" without the drops -> train result 0.9818 -> test result 0.99
#Same results with 4 and 0 poly... -> 0 is faster
#alpha 0.1 and 50000 n_cycle -> training result 0.981875 | alpha 1 and n_cycle 5000 -> training result 0.981875 -> test result 0.99 
#Batch, mini-batch and batch give same result, mini-batch is faster... Stochastic is fastest but does not always give correct answer
#Drop best hand -> traning result 0.89125 -> test result 0.932
#Arithmancy drop gives same result
#Astronomy drop -> test result 0.665
#Herbology drop -> test result 0.907
#Defense against the dark arts drop -> 0.652
#Divination drop -> 0.968
#Muggle studies drop -> train  0.97 -> test 0.492
#Ancient RUnes drop -> train 0.974 -> test 0.945
#History of Magic drop -> train 0.97 -> test 0.988
#Transfiguration drop -> test 0.565
#Potions drop -> train 0.89 -> test 0.985
#Care of Magical Creatures drop -> train 0.974 -> test 0.973
#Charms drop -> train 0.981875 -> test 0.988
#Flying drop -> train 0.973 -> test 0.968
