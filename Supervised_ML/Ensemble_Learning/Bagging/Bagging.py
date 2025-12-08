
import numpy as np

class BaggingClassifier:

    '''1. creating bagging classifier class 
        - it takes a base model ( weak learner), number of base classifiers'''
    def __init__(self, base_classifier, n_estimators):
        self.base_classifier = base_classifier
        self.n_estimators = n_estimators
        self.classifiers = []




    '''2. Bootstrap Sampling

        For each estimator:

        Perform bootstrap sampling with replacement from training data.
        Train a fresh instance of the base classifier on sampled data.
        Save the trained classifier in the list. '''
    def fit(self, X, y):
        for _ in range(self.n_estimators):
            indices = np.random.choice(len(X), len(X), replace=True)
            X_sampled, y_sampled = X[indices], y[indices]
            clf = self.base_classifier.__class__()
            clf.fit(X_sampled, y_sampled)
            self.classifiers.append(clf)
        return self.classifiers
    


    
    '''3.  Implement the predict Method Using Majority Voting

        Collect predictions from each trained classifier.

        Use majority voting across all classifiers to determine final prediction.'''
    def predict(self, X):
        predictions = np.array([clf.predict(X) for clf in self.classifiers])
        majority_votes = np.apply_along_axis(
            lambda x: np.bincount(x).argmax(), axis=0, arr=predictions)
        return majority_votes
    