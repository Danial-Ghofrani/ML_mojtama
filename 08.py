# import numpy as np
#
#
# def gini_impurity(y):
#     _, count = np.unique(y, return_counts=True)
#     probabilities = count / count.sun()
#
#     gini = 1 - np.


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report
import numpy as np


iris = load_iris()


X = iris.data
y = iris.target
feature_labels = iris.feature_names
class_labels = iris.target_names

samples, features = X.shape
classes = np.unique(X)
print(feature_labels)
print(class_labels)


x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=32)


clf = DecisionTreeClassifier(min_samples_leaf=6, max_depth=3)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
print(classification_report(y_test, y_pred))


plt.figure(figsize=(20, 10))
plot_tree(clf, feature_names = iris.feature_names, class_names=class_labels, rounded=True, filled=True, fontsize=10)

plt.show()

node_indicator = clf.decision_path(x_train)

for node_id in node_indicator:
    print(node_id)










def target_generator(x):
    if x%2 == 0 and x%3 == 0:
        return "six"

    elif x% 5 ==0:
        return "five"

    else:
        return "other"




X = np.arange(1, 101)
y_labels = target_generator(X)

encoder = Labe