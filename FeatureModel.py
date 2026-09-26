"""
TODO: header
"""

from typing import Dict, Tuple

import numpy as np
from numpy.typing import NDArray

from Partition import Example, Partition

ConfusionMatrix = NDArray[np.int64]

################################################################################
# CLASSES
################################################################################

class FeatureModel:

    def __init__(self, partition: Partition, feature: str) -> None:
        """
        The contructor takes a partition (Partition of the training dataset) and
        a feature (string) which is the sole feature that will be used for
        predictions.
        """
        self.feature: str = feature
        self.probs: Dict[str, float] = {}

        features = partition.F
        values = features.get(feature)
        counts = {}
        for v in values:
            counts[v] = [0,0] #[label = class, number times it appeared]

        #for each data point, get value of specific feature and class
        for example in partition.data:
            val = example.features.get(feature)
            counts[val][1] += 1
            if example.label == 1:
                counts[val][0] += 1

        #calculate probs
        for v in values:
            if counts.get(v)[1] == 0:
                self.probs[v] = 0
                #print("Flag: Zero examples of " + str(v) + " for feature " + str(feature) + " in data. p assumes 0.")
            else: 
                self.probs[v] = counts.get(v)[0] / counts.get(v)[1]


    def classify(self, example: Example, threshold: float) -> int:
        """
        This helper method classifies one example (Example from the test
        dataset) as -1 or 1 using the given threshold.
        """
        v = example.features.get(self.feature)
        if self.probs.get(v) >= threshold:
            return 1
        else:
            return -1 
      

    def classify_all(
        self, partition: Partition, threshold: float
    ) -> Tuple[ConfusionMatrix, int]:
        """Return the confusion matrix and number of correct predictions."""
        # TODO: Classify every example in partition.
        
        TP, TN, FP, FN = 0, 0, 0, 0

        for i in partition.data:
            true_label = i.label
            predicted_label = self.classify(i, threshold)
            if predicted_label == 1 and true_label == 1:
                TP += 1
            elif predicted_label == 1 and true_label == -1:
                FP += 1
            elif predicted_label == -1 and true_label == -1:
                TN += 1
            elif predicted_label == -1 and true_label == 1:
                FN += 1

        
        print("feature: " + str(self.feature) + ", threshold: " + str(threshold) + "\n")
        #Confusion Matrix [TN FP][FN TP]
        ConfusionMatrix = [[TN, FP], [FN, TP]]

        print("          predicted")
        print("            -1   1")
        print("          -----------")
        print("actual -1 | " + str(TN) +  "  " + str(FP))
        print("        1 | " + str(FN) +  "  " + str(TP) + "\n")
        
        accuracy = (TP  + TN) / (TP + TN + FP + FN)
        print("accuracy: " + str(round(accuracy, 3)) + " (" + str(TP+TN) + " out of " + str(TP + TN + FP + FN) + " correct)")
        
        if (FP + TN) == 0:
            #print("FP + TN is 0, FPR is 0%")
            false_positive_rate = 0
        else: 
            false_positive_rate = FP / (FP + TN)
        
        if (TP + FN) == 0:
            #print("TP + FN is 0, TPR is 0")
            true_positive_rate = 0
        else: 
            true_positive_rate = TP / (TP + FN)

        print("false-positive rate: "+ str(round(false_positive_rate, 3)))
        print("true-positive rate: "+ str(round(true_positive_rate, 3)) + "\n")

        return ConfusionMatrix, (TP + TN)

        

        
        


        

################################################################################
# MAIN
################################################################################

def main() -> None:
    # TODO: test your class here
    pass

if __name__ == "__main__":
    main()
