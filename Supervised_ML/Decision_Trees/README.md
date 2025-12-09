# Decision Tree Implementation Overview

This file implements a **Decision Tree Classifier** from scratch. Here's what each part does:

## **Node Class**
A simple container representing a single node in the tree. Each node stores:
- `feature` / `threshold` — the split rule (if feature ≤ threshold, go left; else go right)
- `left` / `right` — child nodes
- `gain` — information gain from this split
- `value` — the predicted class (only set for leaf nodes)
---

## **DecisionTree Class Methods**

### `__init__(min_samples, max_depth)`
Sets stopping conditions:
- Stop splitting if a node has fewer than `min_samples` samples
- Stop splitting if tree reaches `max_depth`

### `split_data(dataset, feature, threshold)`
Partitions data into two groups based on a feature and threshold value. Simple: if `row[feature] <= threshold`, it goes left; otherwise right.

### `entropy(y)`
Measures **disorder** in labels using the entropy formula: $H = -\sum p_i \log_2(p_i)$
- High entropy = mixed labels (uncertain)
- Low entropy = pure labels (confident)

### `information_gain(parent, left, right)`
Calculates how much **purity improves** after a split:
$$\text{IG} = H(\text{parent}) - \left( \frac{|L|}{|P|} H(L) + \frac{|R|}{|P|} H(R) \right)$$
Higher gain = better split.

### `best_split(dataset, num_samples, num_features)`
**The core of tree building.** Tries every feature and every unique value as a threshold, calculates information gain for each, and returns the split with the highest gain.

### `calculate_leaf_value(y)`
When you reach a leaf, predict the **most common class** in that leaf.

### `build_tree(dataset, current_depth)`
**Recursively builds the tree:**
1. Check stopping conditions (min samples, max depth)
2. If can split: find best split, recursively build left and right subtrees
3. If can't split: create a leaf node with the most common class

### `fit(X, y)`
Entry point: combines features (X) and labels (y), then calls `build_tree()` to train.

### `predict(X)` & `make_prediction(x, node)`
**Uses the trained tree to classify new data:**
- For each sample, start at root and traverse: if `feature ≤ threshold` go left, else go right
- Stop at leaf and return its predicted class

---

## **How It Works (End-to-End)**

1. **Fit** the tree with training data → builds decision rules via information gain
2. **Traverse** new samples through the rules → reaches leaf → returns prediction

The tree greedily picks the split that most reduces label entropy at each node, stopping when it's deep or small enough.