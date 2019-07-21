# -*- coding: utf-8 -*-

"""Console script for abalone_dpacia_sklearn."""
import sys
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from .abalone_dpacia_sklearn import Abalone


def main(args=None):
    """Console script for abalone_dpacia_sklearn."""
    aba = Abalone()
    abalone_dataset = aba.load()
    xtrain, xtest, ytrain, ytest = aba.split(abalone_dataset)

    plt.figure()

    plt.plot0 = aba.decisiontreeregressor(xtrain,xtest,ytrain,ytest)
    plt.plot1 = aba.svr(xtrain,ytrain,xtest, ytest)

    plt.show()


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
