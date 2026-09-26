"""
Command line and file reading utils for Lab 4.
Authors: Sara Mathieson + Allison Gong
Date: 2/9/2025
"""

from collections import OrderedDict
import argparse
from argparse import Namespace
from typing import List

# our imports
from Partition import Example, FeatureValues, Partition

def parse_args() -> Namespace:
    """Parse command line arguments (train and test arff files)."""
    parser = argparse.ArgumentParser(prog="run ROC",
        description='run ROC curve evaluation on test dataset')

    parser.add_argument('-r', '--train_filename', type=str, required=True,
        help='path to train arff file')
    parser.add_argument('-e', '--test_filename',  type=str, required=True,
        help='path to test arff file')

    return parser.parse_args()

def read_arff(filename: str) -> Partition:
    """Read arff file into Partition format."""
    arff_file = open(filename, 'r', encoding='utf-8')
    data: List[Example] = []
    F: FeatureValues = OrderedDict()

    header = arff_file.readline()
    line = arff_file.readline().strip()

    # read the attributes
    while line != "@data":
        line = line.replace('{','').replace('}','').replace(',','')
        tokens = line.split()
        name = tokens[1][1:-1]
        features = tokens[2:]

        # label
        if name != "class":
            F[name] = features
        else:
            first = tokens[2]
        line = arff_file.readline().strip()

    # read the examples
    for line in arff_file:
        tokens = line.strip().split(",")
        X_dict = {}
        i = 0
        for key in F:
            val = tokens[i]
            X_dict[key] = val
            i += 1
        label = -1 if tokens[-1] == "e" else 1 # 'e'=edible, 'p'=poisonous
        data.append(Example(X_dict, label))

    arff_file.close()

    partition = Partition(data, F)
    return partition
