import pandas as pd
import matplotlib.pyplot as mpl
import seaborn as sb


# X-axis from left to right and y_axis from top to bottom:
# [0'Arithmancy', 1'Astronomy', 2'Herbology', 3'Defense Against the Dark Arts', 4'Divination',
#5'Muggle Studies', 6'Ancient Runes', 7'History of Magic', 8'Transfiguration', 9'Potions',
#10'Care of Magical Creatures', 11'Charms',12'Flying']
#sb.pairplot(data, hue="Hogwarts House") -> One huge pairplot

if __name__ == "__main__":
    data = pd.read_csv("datasets/dataset_train.csv", index_col=0)
    data = data.drop(["First Name", "Last Name", "Birthday", "Best Hand"], axis=1)
    courses = data.columns[1:]
    for course1 in courses:
        for course2 in courses:
            if course1 != course2:
                sb.pairplot(data, hue="Hogwarts House", vars=[course1, course2])
                mpl.show()

#Pairplots plot one variable with a histogram and otherwise two variables with a scatter plot
#This is useful to find correlations between two variables, for example if points on scatterplot are moving
#upwards while moving to the right, the two variables are positively correlated
#Two variables that are strongly correlated are not interesting for the AI, and one of them can be eliminated
#You can also see if some features are the same, homogenous
#You can also see and eliminate data with little variation as it does not have a lot of prediction power

#From this visualization, what features are you going to use for your logistic regression?
#Features that are not strongly correlated, being homogenous, positively or negatively correlated.

#Final features conclusions:
#Arithmacy and care of magic creatures are homogenous across the variables and can be eliminated(houses)
#Astronomy and Defense against the dark arts are completely positively correlated, this means we can eliminate one of them.

#->Arithmancy, care of magic creatures and Astronomy are the final eliminated features
