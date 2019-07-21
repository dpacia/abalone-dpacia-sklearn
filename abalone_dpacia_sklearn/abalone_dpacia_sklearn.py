import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
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

    def split(self, abadata):
        """
        Splits the Abalone dataset into 80/20 sets

        :param abadata:
        :return:
        """
        y = abadata.rings.values
        del abadata["rings"]
        XTrain, XTest, yTrain, yTest = train_test_split(abadata, y, random_state=0, train_size=0.2)
        return XTrain, XTest, yTrain, yTest


    def decisiontreeregressor(self, XTrain, XTest, yTrain, yTest):
        """Peforms Decision Tree Regression on the training dataset, generates
        quality metrics and scatter plot between predicted (test) and true values.
        """

        model = DecisionTreeRegressor( )

        model.fit(XTrain, yTrain)
        # apply the model to the test and training data
        predicted_test_y = model.predict(XTest)
        predicted_train_y = model.predict(XTrain)

        # print classification report, accuracy_score, and mean_squared_error

        print(classification_report(yTest, predicted_test_y))
        print("Decision Tree Regressor: ")
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

        #plt.show()

    def SVR(self, XTrain, yTrain, XTest):
        """ Implements support vector regression with linear, polynomial and radial basis
         function kernels.

         Based on code from Scikit-learn documentation
         https://scikit-learn.org/stable/auto_examples/svm/plot_svm_regression.html#sphx-glr-download-auto-examples-svm-plot-svm-regression-py
         """

        # #############################################################################
        # Fit regression model
        svr_rbf = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
        svr_lin = SVR(kernel='linear', C=100, gamma='auto')
        svr_poly = SVR(kernel='poly', C=100, gamma='auto', degree=3, epsilon=.1,
                       coef0=1)

        # #############################################################################
        # Look at the results
        lw = 2

        svrs = [svr_rbf, svr_lin, svr_poly]
        kernel_label = ['RBF', 'Linear', 'Polynomial']
        model_color = ['m', 'c', 'g']

        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 10), sharey='all')
        for ix, svr in enumerate(svrs):
            axes[ix].plot(X, svr.fit(XTrain, yTrain).predict(XTest), color=model_color[ix], lw=lw,
                          label='{} model'.format(kernel_label[ix]))
            axes[ix].scatter(X[svr.support_], y[svr.support_], facecolor="none",
                             edgecolor=model_color[ix], s=50,
                             label='{} support vectors'.format(kernel_label[ix]))
            axes[ix].scatter(X[np.setdiff1d(np.arange(len(X)), svr.support_)],
                             y[np.setdiff1d(np.arange(len(X)), svr.support_)],
                             facecolor="none", edgecolor="k", s=50,
                             label='other training data')
            axes[ix].legend(loc='upper center', bbox_to_anchor=(0.5, 1.1),
                            ncol=1, fancybox=True, shadow=True)

        fig.text(0.5, 0.04, 'data', ha='center', va='center')
        fig.text(0.06, 0.5, 'target', ha='center', va='center', rotation='vertical')
        fig.suptitle("Support Vector Regression", fontsize=14)





