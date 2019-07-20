import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from collections import Counter


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
        abadata = pd.read_csv("./data/abalone.data", names=column_names)

        # convert sex into factor, remove sex column.
        for label in "MFI":
            abadata[label] = abadata["sex"] == label
        del abadata["sex"]

        return abadata

    def train(self, abadata):
        """
        Splits the Abalone dataset into 80/20 sets

        :param abadata:
        :return:
        """
        y = abadata.rings.values
        del abadata["rings"]
        XTrain, XTest, yTrain, yTest = train_test_split(abadata, y, random_state=0, train_size=0.2)
        model = DecisionTreeRegressor( )

        model.fit(XTrain, yTrain)
        # apply the model to the test and training data
        predicted_test_y = model.predict(XTest)
        predicted_train_y = model.predict(XTrain)

        # print classification report, accuracy_score, and mean_squared_error

        print(classification_report(yTest, predicted_test_y))
        print("accuracy score: %.3f " % accuracy_score(yTest, predicted_test_y))
        print("mean squared error: %.3f " % mean_squared_error(yTest, predicted_test_y))

        # count the occurrences of each point
        c = Counter(zip(yTest, predicted_test_y))
        # create a list of the sizes, here multiplied by 3 for scale
        pt_size_lst = [3 * c[(xx, yy)] for xx, yy in zip(yTest, predicted_test_y)]

        # Plot the results
        plt.figure()
        plt.scatter(yTest, predicted_test_y, color="darkred", s=pt_size_lst)

        plt.xlabel('True number of rings')
        plt.ylabel('Predicted number of rings')
        plt.title('Decision Tree Regressor - Test Data')

        plt.show()





