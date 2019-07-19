import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt


class Abalone():
    """
    Class to load Abalone dataset, train, test, and visualize results with the Scikit-Learn library.

    """

    def __init__(self):
        """Initialize class. """
        pass

    def __call__(self):
        """When called, """
        pass

    def load(self):
        """
        Loads the Abalone dataset from the /data directory
        :return: pd dataframe with Abalone data.
        """
        column_names = ["sex", "length", "diameter", "height", "whole_weight", "shucked_weight",
                        "viscera_weight", "shell_weight", "rings"]
        abadata = pd.read_csv("../data/abalone.data", names=column_names)

        return abadata

    def train(self, abadata):
        """
        Splits the Abalone dataset into 80/20 sets

        :param abadata:
        :return:
        """
        XTrain, XTest, yTrain, yTest = train_test_split(abadata, y, random_state=0, train_size=0.2)
        model = DecisionTreeRegressor()

        model.fit(XTrain, yTrain)
        # apply the model to the test and training data
        predicted_test_y = model.predict(XTest)
        predicted_train_y = model.predict(XTrain)

        # Plot the results
        plt.figure()
        plt.scatter(X, y, s=20, edgecolor="black",
                    c="darkorange", label="data")
        plt.plot(X_test, y_1, color="cornflowerblue",
                 label="max_depth=2", linewidth=2)
        plt.plot(X_test, y_2, color="yellowgreen", label="max_depth=5", linewidth=2)
        plt.xlabel("data")
        plt.ylabel("target")
        plt.title("Decision Tree Regression")
        plt.legend()
        plt.show()





