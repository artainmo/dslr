import numpy as np
import matplotlib.pyplot as mpl
import copy


def l2_regularization(theta):
    theta[0] = 0
    return theta.transpose().dot(theta)

def add_intercept(x_values):
    return np.column_stack((np.full(x_values.shape[0], 1), x_values))

def theta_0(theta):
    theta[0] = 0
    return theta

#Also called the activation function, this is the function that transforms values into probabilities between 0 and 1
def sigmoid_(x):
    return np.divide(1, np.add(1, (np.exp((np.multiply(x, -1)).astype(np.float)))))

def regularized_logistic_gradient(expected_values, x_values, theta, lambda_):
    lenght = x_values.shape[0]
    x_values = add_intercept(x_values)
    res = np.add(x_values.transpose().dot(np.subtract(sigmoid_(x_values.dot(theta)), expected_values)), np.multiply(lambda_, theta_0(copy.deepcopy(theta))))
    return res / lenght

#The correct class needs to be represented numerically with 1 or "yes", while the wrong answer needs to represented as 0
#When doing multiclassification, you need to let each class clasify against all the others
#and find the one with the highest probablity,
#to do that you need to set the class you want as 1 or "yes" and all other the other classes as 0 or "no"
def descriminate_classes(expected_values, class_):
    i = 0
    y = copy.deepcopy(expected_values)
    while i < expected_values.shape[0]:
        if y[i] == class_:
            y[i] = 1
        else:
            y[i] = 0
        i += 1
    return y

def get_mini_batch(x_values, expected_values, b):
    last = 0
    counter = 0
    while True:
        for i in range(0,x_values.shape[0],b):
            yield (x_values[last:i], expected_values[last:i], counter*x_values.shape[0] + i)
            last = i
        counter += 1

def shuffle_(x_values, expected_values):
    shuffle = np.column_stack((x_values, expected_values))
    np.random.shuffle(shuffle)
    shuffle = shuffle.transpose()
    return ((shuffle[:-1]).transpose(), (shuffle[-1:]).transpose())

#Logistic regression is used as a classification algorithm it gives a binary answer, yes or no
#Logistic regression uses similar techniques as linear regression, its predictions are probabilities between 0 and 1
#If closer to 0 the answer is no and yes if closer to one
class LogisticRegression():
    #b is equal to the mini-batch size, a number between 2 and 32 is recommended
    def __init__(self, type="batch", thetas=[0,0], alpha=0.0001, n_cycle=1000000, lambda_=0, b=18):
        if type == "batch" or type == "mini_batch" or type == "stochastic":
            self.type = type
            self.theta = thetas
            self.alpha = alpha
            self.n_cycle = n_cycle
            self.lambda_ = lambda_
            self.b = b
        else:
            print("Wrong gradient descend type")
            exit()


    #Each gradient descend step uses all x_values(rows), one n_cycle equals one step
    #Test all methods to see what seems best
    def batch_(self, x_values, expected_values):
        for n in range(0,self.n_cycle):
            gradient = regularized_logistic_gradient(expected_values, x_values, self.theta, self.lambda_)
            self.theta = np.subtract(self.theta, (self.alpha*gradient))
        return self.theta

    #Each gradient descend step uses b x_values(rows), or a part of the whole set of x_values, one n_cycle goes through the b rows
    #Is a compromise between batch and stochastic gradient descend
    #that tries to avoid local minima while not converging too slow by letting you choose how much of the data you use on huge datasets
    def mini_batch_(self, x_values, expected_values):
        for x, y, i in get_mini_batch(x_values, expected_values, self.b):
            if i > self.n_cycle:
                break
            gradient = regularized_logistic_gradient(expected_values, x_values, self.theta, self.lambda_)
            self.theta = np.subtract(self.theta, (self.alpha*gradient))
        return self.theta

    #One n_cycle goes through one row
    #Is great at avoiding local minima, can be way faster for large datasets
    def stochastic_(self, x_values, expected_values):
        i = 0
        x_set, y_set = shuffle_(x_values, expected_values)
        while i < x_set.shape[0]:
            if i > self.n_cycle:
                break
            gradient = regularized_logistic_gradient(np.array([y_set[i]]), np.array([x_set[i]]), self.theta, self.lambda_)
            self.theta = np.subtract(self.theta, (self.alpha*gradient))
            i += 1
        return self.theta

    def fit_(self, x_values, expected_values):
        if self.type == "batch":
            return self.batch_(x_values, expected_values)
        elif self.type == "mini_batch":
            return self.mini_batch_(x_values, expected_values)
        elif self.type == "stochastic":
            return self.stochastic_(x_values, expected_values)



    def cost_(self, predicted_values, expected_values):
        one = np.ones(predicted_values.shape)
        summation = expected_values.transpose().dot((np.log(predicted_values))) + (np.subtract(one, expected_values).transpose().dot((np.log(np.subtract(one, predicted_values)))))
        summation = (summation / predicted_values.shape[0] * -1)
        return (summation + np.multiply((self.lambda_ / (2 * predicted_values.shape[0])), l2_regularization(copy.deepcopy(self.theta))))[0][0]

    #Difference with linear regression is that the prediction values are transformed into probablities between one and zero with the sigmoid function
    def predict_(self, input_variables):
        predicted_values = []
        input_variables = add_intercept(input_variables)
        return sigmoid_(input_variables.dot(self.theta))

    def plot_(self, x_values, expected_values):
        predicted_values = self.predict_(x_values)
        cost = self.cost_(predicted_values, expected_values)
        mpl.plot(x_values, predicted_values, color="orange")
        mpl.plot(x_values, expected_values, linestyle="",marker="o", color="blue")
        mpl.title("Cost: " + str(cost))
        mpl.show()
