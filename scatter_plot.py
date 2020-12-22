import sys
from utils.MyStats import *

if __name__=="__main__":
    if len(sys.argv) != 2:
        print("Error: number arguments")
        exit()
    classes_ = ["Ravenclaw", "Slytherin", "Gryffindor", "Hufflepuff"]
    path = sys.argv[1]
    classes_label = "Hogwarts House"
    drop_columns = ["First Name", "Last Name", "Birthday", "Best Hand"] #Drop all non-numerical columns
    scatter_plot(path, classes_, classes_label, drop_columns)

#What are the two features that are similar ?
#Care of magic creatures and arithmancy
#To reduce the computational cost and complexity of the algorithm we can elimiate one of similar features
