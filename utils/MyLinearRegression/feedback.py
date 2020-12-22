import sys
import numpy as np
import pandas as pd
from .logistic_regression import *
from .logistic_metrics import *
from .handle_data import *
import sklearn.metrics

def get_class_probabilities(classes, x_values, thetas):
    for class_ in classes:
        theta = np.array(thetas[class_])
        LR = LogisticRegression(thetas=theta)
        try:
            probabilities = np.column_stack((probabilities, LR.predict(x_values)))
        except:
            probabilities = LR.predict(x_values)
    return probabilities

def probabilities_to_answer(classes, probabilities):
    answer = np.array([])
    for row in probabilities:
        max = np.NINF
        for i, probs in enumerate(row):
            if probs > max:
                max = probs
                class_ = classes[i]
        answer = np.append(answer, class_)
    return answer

def class_answers(classes, x_values, thetas):
    return probabilities_to_answer(classes, get_class_probabilities(classes, x_values, thetas))


def feedback(LR, x_values, y):
    predict = LR.predict(x_values)
    print("cost: " + str(LR.cost(predict, y)))
    print("accuracy: " + str(accuracy_score(predict, y)))


def sklearn_feedback(your_answers, real_answers):
    print("accuracy: " + str(sklearn_accuracy_score(your_answers, real_answers)))
    print("sklearn accuracy: " + str(sklearn.metrics.accuracy_score(your_answers, real_answers)))
