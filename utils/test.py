import sys
import numpy as np
import pandas as pd
from utils.logistic_regression import *
from utils.logistic_metrics import *
from utils.handle_data import *
import sklearn.metrics

houses = np.array(["Ravenclaw", "Slytherin", "Gryffindor", "Hufflepuff"])


def get_house_probabilities(x_values):
    for house in houses:
        theta = np.array(pd.read_csv("datasets/theta.csv")[[house]])
        LR = LogisticRegression(thetas=theta)
        predict = LR.predict_(x_values)
        try:
            probabilities = np.column_stack((probabilities, predict))
        except:
            probabilities = predict
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


def binary_feedback(LR, x_values, y):
    predict = LR.predict_(x_values)
    print("cost: " + str(LR.cost_(predict, y)))
    print("accuracy: " + str(accuracy_score_(predict, y, binary=True)))


def final_feedback(x_values, y):
    print("Final test with test set")
    probabilities = get_house_probabilities(x_values)
    answer = house_probabilities_to_answer(probabilities)
    print("accuracy: " + str(accuracy_score_(answer, y, binary=False)))
    print("sklearn accuracy: " + str(sklearn.metrics.accuracy_score(answer, y)))


def assemble_thetas(thetas, theta):
    try:
        thetas = np.column_stack((thetas, theta))
    except:
        thetas = theta
    return thetas
