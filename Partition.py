"""
Partition class (holds feature information, feature values, and labels for a
dataset). Includes helper class Example.
Author: Sara Mathieson
Date: 8/26/2021
"""

from typing import Dict, List

FeatureMap = Dict[str, str]
FeatureValues = Dict[str, List[str]]

################################################################################
# CLASSES
################################################################################

class Example:

    def __init__(self, features: FeatureMap, label: int) -> None:
        """Helper class (like a struct) that stores info about each example."""
        # dictionary. key=feature name: value=feature value for this example
        self.features = features
        self.label = label # in {-1, 1}

class Partition:

    def __init__(self, data: List[Example], F: FeatureValues) -> None:
        """Store information about a dataset"""
        # list of Examples
        self.data = data
        self.n = len(self.data)

        # dictionary. key=feature name: value=list of possible values
        self.F = F
