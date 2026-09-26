"""
TODO: header
"""

from typing import List, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from Partition import Partition
from util import parse_args, read_arff
from FeatureModel import FeatureModel

ConfusionMatrix = NDArray[np.int64]

################################################################################
# MAIN
################################################################################

def main() -> None:
    
    args = parse_args()
    train = read_arff(args.train_filename)
    test = read_arff(args.test_filename) 

    # TODO: for each feature, call create_roc
    print("training examples: " + str(train.n))
    print("testing examples: " + str(test.n) + "\n")

    for f in train.F:
        create_roc(train, test, f)


################################################################################
# HELPER FUNCTIONS
################################################################################

def confusion_matrix(
    train_partition: Partition,
    test_partition: Partition,
    feature: str,
    threshold: float,
) -> ConfusionMatrix:
    """Return a 2-by-2 confusion matrix for one feature and threshold."""
    model = FeatureModel(train_partition, feature)
    return model.classify_all(test_partition, threshold)
 
def false_and_true_positive_rates(
    confusion: ConfusionMatrix,
) -> Tuple[float, float]:
    """Return (false-positive rate, true-positive rate)."""
    # TODO: Calculate both rates from confusion.
    raise NotImplementedError


def approximate_auc(
    false_positive_rates: Sequence[float],
    true_positive_rates: Sequence[float],
) -> float:
    """Approximate AUC with trapezoids after ordering points by FPR."""
    # TODO: Sort the points and implement the trapezoid rule yourself.
    raise NotImplementedError


def create_roc(
    train_partition: Partition, test_partition: Partition, feature: str
) -> Tuple[List[float], List[float]]:

    thresholds = [0.5]
    confusion_matrices = list()
    for threshold in thresholds:
        confusion_matrices.append(confusion_matrix(train_partition, test_partition, feature, threshold))
    

    # TODO: plot the ROC curve using plt.plot


if __name__ == "__main__":
    main()
