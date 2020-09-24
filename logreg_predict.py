import sys
import numpy as np
import pandas as pd
from utils.logistic_regression import *
from utils.logistic_metrics import *
from utils.handle_data import *

houses = np.array(["Ravenclaw", "Slytherin", "Gryffindor", "Hufflepuff"])


def get_house_probabilities(x_values):
    for house in houses:
        theta = np.array(pd.read_csv("datasets/" + sys.argv[2])[[house]])
        LR = LogisticRegression(thetas=theta)
        try:
            probabilities = np.column_stack((probabilities, LR.predict_(x_values)))
        except:
            probabilities = LR.predict_(x_values)
    return probabilities

def house_probabilities_to_answer(probabilities):
    answer = np.array([])
    for row in probabilities:
        max = np.NINF
        for i, probs in enumerate(row):
            if probs > max:
                max = probs
                house = houses[i]
        answer = np.append(answer, house)
    return answer

def main():
    x_values = pd.read_csv("datasets/" + sys.argv[1], index_col=0)
    x_values = np.array(x_values.drop(["Hogwarts House", "First Name", "Last Name", "Birthday", "Best Hand", "Arithmancy", "Care of Magical Creatures", "Astronomy"], axis=1))
    x_values = NAN_to_median(x_values)
    x_values = minmax_normalization(x_values)
    x_values = add_polynomial_features(x_values, 2)
    probabilities = get_house_probabilities(x_values)
    answer = house_probabilities_to_answer(probabilities)
    pd.DataFrame(answer, columns=["answers"]).to_csv("datasets/houses.csv")


if __name__=="__main__":
    main()
