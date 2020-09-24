import pandas as pd
import matplotlib.pyplot as mpl

houses_ = ["Ravenclaw", "Slytherin", "Gryffindor", "Hufflepuff"]

#Use this function to avoid making the same comparisions
def only_after_in_list_check(courses, curr_course, check):
    for course in courses:
        if check == course:
            return False
        if curr_course == course:
            return True

def get_data_houses():
    house_data = []
    data = pd.read_csv("datasets/dataset_train.csv", index_col=0)
    data = data.drop(["First Name", "Last Name", "Birthday", "Best Hand"], axis=1)
    for house in houses_:
        house_data.append(data[data["Hogwarts House"] == house])
    return house_data

#Only way to compare if two features are same is by comparing them in groups of two, otherwise because of one very large dataset, small dataset becomes not readable
#This makes a total of 13! different possibilities that is why showing them one by one
if __name__ == "__main__":
    houses = get_data_houses()
    courses = houses[0].columns[1:]
    for course1 in courses:
        for course2 in courses:
            if only_after_in_list_check(courses, course1, course2):
                figure = houses[0].plot.scatter(y=course1, x=course2, s=[2])
                for i, color in zip(range(1,4), ["c", "g", "r"]):
                    houses[i].plot.scatter(y=course1, x=course2, s=[2], c=color, ax=figure)
                mpl.show()

#What are the two features that are similar ?
#Care of magic creatures and arithmacy -> This means they can be elimiated, same comclusion as with the histogram
#To reduce the computational cost and complexity of the algorithm we can elimiate similar features
