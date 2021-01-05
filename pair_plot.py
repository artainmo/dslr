import sys
from utils.MyStats import *

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Error: number arguments")
        exit()
    path = sys.argv[1]
    class_label = "Hogwarts House"
    drop_columns = ["First Name", "Last Name", "Birthday", "Best Hand"]
    pair_plot(path, class_label, drop_columns)


#Pairplots compares two features over the different classes, in a line plot and scatterplot

#Scatterplots are useful to find correlations and homogenousity between two features.
#If one of two features are the same, one of them is not interesting for AI and can be eliminated.

#Line plots are useful to find correlations between classes in one feature
#Features that are homogenous or have low variation over the classes are not interesting for AI neither as they have low predictive power

#From this visualization, what features are you going to use for your logistic regression?

#Features: "Arithmancy and care of magic creatures" are homogenous across the classes and at least one can be eliminated

#->Arithmancy is the final eliminated feature
#Eliminating features speeds up the algorithm
