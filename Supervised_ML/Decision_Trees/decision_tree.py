import numpy as np

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature        # index of feature to split on
        self.threshold = threshold    # numeric split point
        self.left = left              # left child node
        self.right = right            # right child node
        self.value = value            # class label if leaf


class DecisionTree:
    def __init__(self, max_depth=5, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    # ----- Impurity (Gini) -----
    def gini(self, y):
        classes, counts = np.unique(y, return_counts=True)
        p = counts / len(y)
        return 1 - np.sum(p**2)

    # ----- Find the best split -----
    def best_split(self, X, y):
        n_samples, n_features = X.shape
        best_feature, best_threshold = None, None
        best_impurity = float("inf")

        for feature in range(n_features):
            values = np.unique(X[:, feature])

            # Try possible thresholds
            for threshold in values:
                left_idx = X[:, feature] < threshold
                right_idx = ~left_idx

                if left_idx.sum() == 0 or right_idx.sum() == 0:
                    continue

                # Weighted impurity
                left_imp = self.gini(y[left_idx])
                right_imp = self.gini(y[right_idx])
                impurity = (len(y[left_idx]) / len(y)) * left_imp + \
                           (len(y[right_idx]) / len(y)) * right_imp

                if impurity < best_impurity:
                    best_impurity = impurity
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    # ----- Build tree recursively -----
    def build(self, X, y, depth=0):
        # stopping conditions
        if (depth >= self.max_depth or
            len(y) < self.min_samples_split or
            len(np.unique(y)) == 1):

            leaf_value = self.most_common(y)
            return Node(value=leaf_value)

        feature, threshold = self.best_split(X, y)

        if feature is None:
            leaf_value = self.most_common(y)
            return Node(value=leaf_value)

        left_idx = X[:, feature] < threshold
        right_idx = ~left_idx

        left = self.build(X[left_idx], y[left_idx], depth+1)
        right = self.build(X[right_idx], y[right_idx], depth+1)

        return Node(feature, threshold, left, right)

    # ----- Helper -----
    def most_common(self, y):
        values, counts = np.unique(y, return_counts=True)
        return values[np.argmax(counts)]

    # ----- Public API -----
    def fit(self, X, y):
        self.root = self.build(X, y)

    def predict_one(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature] < node.threshold:
            return self.predict_one(x, node.left)
        else:
            return self.predict_one(x, node.right)

    def predict(self, X):
        return np.array([self.predict_one(x, self.root) for x in X])
