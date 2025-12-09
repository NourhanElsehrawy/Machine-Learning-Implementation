import os 
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sklearn.tree import DecisionTreeClassifier
from Bagging import BaggingClassifier
import numpy as np 


class RandomForestClassifierCustom(BaggingClassifier):
    def __init__(self, n_estimators=100, max_features="sqrt"):
        # Initialize BaggingClassifier with base DecisionTreeClassifier
        super().__init__(base_classifier=DecisionTreeClassifier(), n_estimators=n_estimators)
        self.max_features = max_features

    def fit(self, X, y):
        self.classifiers = []

        for _ in range(self.n_estimators):
            # Bootstrap sampling
            indices = np.random.choice(len(X), len(X), replace=True)
            X_sampled, y_sampled = X[indices], y[indices]

            # Train a decision tree with random features
            clf = DecisionTreeClassifier(max_features=self.max_features)
            clf.fit(X_sampled, y_sampled)
            self.classifiers.append(clf)

        return self  # Return self to follow sklearn style

    def predict(self, X):
        # Use BaggingClassifier's majority voting
        return super().predict(X)
    

