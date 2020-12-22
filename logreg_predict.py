import sys
import numpy as np
import pandas as pd
from utils.MyLinearRegression import *
from logreg_train import get_xy_values

if __name__=="__main__":
    classes = np.array(["Ravenclaw", "Slytherin", "Gryffindor", "Hufflepuff"])
    if len(sys.argv) != 3:
        print("Error: number arguments")
        exit()
    x_values, expected_values = get_xy_values(sys.argv[1], "Hogwarts House")
    answer = class_answers(classes, x_values, pd.read_csv(sys.argv[2]))
    answer_file = pd.DataFrame(answer, columns=["Hogwarts House"])
    answer_file.index.name = "Index"
    try:
        answer_file.to_csv("datasets/houses.csv")
    except:
        print("Error: argument file")
        exit()
