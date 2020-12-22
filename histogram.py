import sys
import pandas as pd
import matplotlib
import matplotlib.pyplot as mpl
#matplotlib.use('TkAgg') #Make matplotlib compatible with Big Sur on mac

def get_data(path, classes_label, classes, drop_columns):
    class_data = []
    try:
    	data = pd.read_csv(path, index_col=0)
    except:
        print("Error: argument file")
        exit()
    data = data.drop(drop_columns, axis=1)
    for class_ in  classes:
        class_data.append(data[data[classes_label] == class_])
    return class_data


def histogram(path, classes_label, classes, drop_columns):
    data = get_data(path, classes_label, classes, drop_columns)
    features = data[0].columns[1:]
    figure,plots = mpl.subplots(5,3)
    plots[4][1].remove()
    plots[4][2].remove()
    for index1, feature in enumerate(features):
        index2 = int(index1) % 3
        index1 = int(index1) // 3
        plots[index1][index2].hist([data[0][feature].to_list(), data[1][feature].to_list(), data[2][feature].to_list(), data[3][feature].to_list()]) #Plots for a certain house and course, grades and number of people that got those grades, one plot per course
        plots[index1][index2].set_title(feature)
    mpl.tight_layout()
    mpl.show()


if __name__=="__main__":
    if len(sys.argv) != 2:
        print("Error: number arguments")
        exit()
    path = sys.argv[1]
    drop_columns = ["First Name", "Last Name", "Birthday", "Best Hand"]
    classes_label = "Hogwarts House"
    classes = ["Ravenclaw", "Slytherin", "Gryffindor", "Hufflepuff"]
    histogram(path, classes_label, classes, drop_columns)

#Histogram y-axis is frequency of x-axis score

#Which Hogwarts course has a homogeneous score distribution between all four houses?
#Care of magic creatures and arithmancy
#Features that are homogeneous over the classes are not intersting and could be eliminated to limit computational costs
