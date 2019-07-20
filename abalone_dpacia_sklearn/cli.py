# -*- coding: utf-8 -*-

"""Console script for abalone_dpacia_sklearn."""
import sys
from .abalone_dpacia_sklearn import Abalone


def main(args=None):
    """Console script for abalone_dpacia_sklearn."""
    aba = Abalone()
    abalone_dataset = aba.load()
    aba.train(abalone_dataset)


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
