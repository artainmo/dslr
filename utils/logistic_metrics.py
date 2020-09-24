import numpy as np
import pandas as pd


def binary_predict(predicted_values):
    i = 0
    while i < predicted_values.shape[0]:
        if predicted_values[i] < 0.5:
            predicted_values[i] = 0
        else:
            predicted_values[i] = 1
        i += 1
    return predicted_values

#There are two types of errors that occur in a classificatio algorithm
#False positives, or predicting "yes", while the expected answer was "no"
#False negatives, or predicting "no", while the expected answer was "yes"
def binary_positives_negatives(expected_values, predicted_values):
    data = {"true_positive": 0,
            "true_negative": 0,
            "false_positive": 0,
            "false_negative": 0}
    predicted_values = binary_predict(predicted_values)
    for expected_value, predicted_value in zip(expected_values, predicted_values):
        if expected_value == predicted_value:
            if expected_value != 1:
                data["true_negative"] += 1
            else:
                data["true_positive"] += 1
        else:
            if predicted_value != 1:
                data["false_negative"] += 1
            else:
                data["false_positive"] += 1
    return data

#Final test for logistic regression multiclassification
def positives_negatives(expected_values, predicted_values):
    correct = 0
    for expected_value, predicted_value in zip(expected_values, predicted_values):
        if expected_value == predicted_value:
            correct += 1
    return correct / expected_values.shape[0]

#The accuracy score gives you the percentage of correct answers
def accuracy_score_(predicted_values, expected_values, binary=True):
    if binary == True:
        data = binary_positives_negatives(expected_values, predicted_values)
        return (data["true_positive"] + data["true_negative"]) / (data["true_positive"] + data["false_positive"] + data["true_negative"] + data["false_negative"])
    else:
        return positives_negatives(expected_values, predicted_values)

#Precision gives you the percentage of correct "yes" answers, it gives feedback about the false positives
def precision_score_(expected_values, predicted_values, class_=1):
    data = positives_negatives(expected_values, predicted_values, class_)
    return (data["true_positive"]) / (data["true_positive"] + data["false_positive"])

#Recall gives you the percentage of Aobject properly classified as class A, it gives feedback about the false negatives.
def recall_score_(expected_values, predicted_values, class_=1):
    data = positives_negatives(expected_values, predicted_values, class_)
    return (data["true_positive"]) / (data["true_positive"] + data["false_negative"])

#f1 is a combination of precision and recall used when trying to optimize both false positives and negatives.
#It is more difficult to optimize for both false positives and negatives, best is to choose what side to optimize for
def f1_score_(predicted_values, expected_values, class_=1):
    precision_score = precision_score_(expected_values, predicted_values, class_)
    recall_score = recall_score_(expected_values, predicted_values, class_)
    return (2 * precision_score * recall_score) / (precision_score + recall_score)




def get_labels(predicted_values):
    labels = []
    for item in predicted_values:
        if item not in labels:
            labels.append(item)
    return labels[::-1]

def comparison(expected_values, predicted_values, label, labels):
    counter = np.zeros(len(labels))
    for expected_value, predicted_value, in zip(expected_values, predicted_values):
        if expected_value == label:
            try:
                i = labels.index(predicted_value)
                counter[i] += 1
            except:
                pass
    return counter

#The confusion matrix gives an overview of both false positives and negatives
#Cost function on its own isn't enough as it does not give information about the false positives and negatives
def confusion_matrix_(expected_values, predicted_values, labels=None, df_option=None):
    confusion_matrix = []
    if labels == None:
        labels = get_labels(predicted_values)
    for label in labels:
         confusion_matrix.append(comparison(expected_values, predicted_values, label, labels))
    if df_option == None:
        return np.array(confusion_matrix)
    else:
        return pd.DataFrame(confusion_matrix, index=labels, columns=labels)
