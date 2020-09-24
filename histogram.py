import pandas as pd
import matplotlib.pyplot as mpl

houses_ = ["Ravenclaw", "Slytherin", "Gryffindor", "Hufflepuff"]

def get_data_houses():
    house_data = []
    data = pd.read_csv("datasets/dataset_train.csv", index_col=0)
    data = data.drop(["First Name", "Last Name", "Birthday", "Best Hand"], axis=1)
    for house in houses_:
        house_data.append(data[data["Hogwarts House"] == house])
    return house_data


if __name__ == "__main__":
    houses = get_data_houses()
    courses = houses[0].columns[1:]
    figure,plots = mpl.subplots(5,3)
    plots[4][1].remove()
    plots[4][2].remove()
    for index1, course in enumerate(courses):
        index2 = int(index1) % 3
        index1 = int(index1) // 3
        plots[index1][index2].hist([houses[0][course].to_list(), houses[1][course].to_list(), houses[2][course].to_list(), houses[3][course].to_list()]) #Plots for a certain house and course, grades and number of people that got those grades, one plot per course
        plots[index1][index2].set_title(course)
    mpl.tight_layout()
    mpl.show()


#Which Hogwarts course has a homogeneous score distribution between all four houses?
#Arithmacy and care of magic creatures
#To reduce the computational cost and complexity of the algorithm we can elimiate features with similar results over the different answer variables(houses)
