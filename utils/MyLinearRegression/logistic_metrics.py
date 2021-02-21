import numpy as np
import pandas as pd

#Set to one or zero
# < 0.5 -> 1
# >= 0.5 -> 0
def binary_set(predicted_values):
    i = 0
    while i < predicted_values.shape[0]:
        if predicted_values[i] < 0.5: #To trade-off precision and recall 0.5 can become 0.7 for example if you want to avoid false positives for cancer
            predicted_values[i] = 0
        else:
            predicted_values[i] = 1
        i += 1
    return predicted_values

#There are two types of errors that occur in a classificatio algorithm
#False positives, or predicting "yes", while the expected answer was "no"
#False negatives, or predicting "no", while the expected answer was "yes"
#Returns false positves, false negatives, true positives and true negatives
def positives_negatives(expected_values, predicted_values, class_=1):
    data = {"true_positive": 0,
            "true_negative": 0,
            "false_positive": 0,
            "false_negative": 0}
    predicted_values = binary_set(predicted_values)
    for expected_value, predicted_value in zip(expected_values, predicted_values):
        if expected_value == predicted_value:
            if expected_value != class_:
                data["true_negative"] += 1
            else:
                data["true_positive"] += 1
        else:
            if predicted_value != class_:
                data["false_negative"] += 1
            else:
                data["false_positive"] += 1
    return data

#The accuracy score gives you the percentage of correct answers
def accuracy_score(predicted_values, expected_values, class_=1):
    data = positives_negatives(expected_values, predicted_values, class_)
    return (data["true_positive"] + data["true_negative"]) / (data["true_positive"] + data["false_positive"] + data["true_negative"] + data["false_negative"])

#Precision gives you the percentage of correct "yes" answers, it gives feedback about the false positives
def precision_score(expected_values, predicted_values, class_=1):
    data = positives_negatives(expected_values, predicted_values, class_)
    return (data["true_positive"]) / (data["true_positive"] + data["false_positive"])

#Recall gives you the percentage of an object properly classified as class A, it gives feedback about the false negatives.
def recall_score(expected_values, predicted_values, class_=1):
    data = positives_negatives(expected_values, predicted_values, class_)
    return (data["true_positive"]) / (data["true_positive"] + data["false_negative"])

#f1 is a combination of precision and recall used when trying to optimize both false positives and negatives.
#It is more difficult to optimize for both false positives and negatives, best is to choose what side to optimize for
def f1_score(predicted_values, expected_values, class_=1):
    precision_score = precision_score_(expected_values, predicted_values, class_)
    recall_score = recall_score_(expected_values, predicted_values, class_)
    return (2 * precision_score * recall_score) / (precision_score + recall_score)

#Returns percentage of correct answers, similar to how the sklearn accuracy function works
def sklearn_accuracy_score(expected_values, predicted_values):
    correct = 0
    if expected_values.shape[0] == 0:
        return "Error: empty";
    for expected_value, predicted_value in zip(expected_values, predicted_values):
        if expected_value == predicted_value:
            correct += 1
    return correct / expected_values.shape[0]

################################################################################CONFUSION MATRIC

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
def confusion_matrix(expected_values, predicted_values, labels=None, df_option=None):
    confusion_matrix = []
    if labels == None:
        labels = get_labels(predicted_values)
    for label in labels:
         confusion_matrix.append(comparison(expected_values, predicted_values, label, labels))
    if df_option == None:
        return np.array(confusion_matrix)
    else:
        return pd.DataFrame(confusion_matrix, index=labels, columns=labels)

################################################################################
