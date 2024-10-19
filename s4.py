import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import time
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import numpy as np
import pandas as pd


### Part One
# X, y = make_blobs(n_samples=100, n_features=2, centers=3, cluster_std=7)
#
#
# # plt.scatter(X[:,0], X[:,1], c=y)
# # plt.show()
#
# fig = plt.figure()
#
# ax = fig.add_subplot(111, projection= "3d")
# ax.scatter(X[:,0], X[:,1], c=y)
#
# plt.show()




#### part two
# X, y = make_blobs(n_samples=100, n_features=2, centers=2, cluster_std=1.5)
#
# clf = SVC(kernel="rbf", C = 0.5, verbose=1)
#
# clf.fit(X,y)
#
# plt.subplot(1,2,1)
# plt.scatter(X[:,0], X[:,1], c=y)
#
# pred = clf.predict(X)
#
# plt.subplot(1,2,2)
# plt.scatter(X[:,0], X[:,1], c=pred)
# plt.show()


# plt.scatter(X[:,0], X[:,1], c=y)
# plt.show()

# fig = plt.figure()
#
# ax = fig.add_subplot(111, projection= "3d")
# ax.scatter(X[:,0], X[:,1], c=y)
#
# plt.show()


#### part three
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import time
X, y = make_blobs(n_samples=100, n_features=2, centers=2, cluster_std=1.5)

svm_start_time = time.time()
svm_clf = SVC()
svm_clf.fit(X,y)
svm_end_time = time.time()
svm_train_time = svm_end_time - svm_start_time

start_time = time.time()
svm_pred = svm_clf.predict(X)
end_time = time.time()
predict_time = end_time - start_time

reg_start_time = time.time()
reg_clf = LogisticRegression()
reg_clf.fit(X, y)
reg_end_time = time.time()
reg_train_time = reg_end_time - reg_start_time

start_time = time.time()
reg_pred = reg_clf.predict(X)
end_time = time.time()
predict_time = end_time - start_time

plt.subplot(1,2,1)
plt.scatter(X[:,0], X[:,1], c=y)

pred = clf.predict(X)

plt.subplot(1,2,2)
plt.scatter(X[:,0], X[:,1], c=pred)
plt.show()


# X, y = make_blobs(n_samples = 100, n_features=2, centers=2, cluster_std=1.2)
#
#
# clf = SVC(kernel="rbf", C=0.5)
#
# clf.fit(X, y)
#
#
# plt.subplot(1,2,2)





# plt.scatter(X[:,0], X[:,1], c=y)
# plt.show()




# fig = plt.figure()
#
# ax = fig.add_subplot(111, projection = 3d)







# params = {"kernel": ["linear", "poly", "rbf", "sigmoid"], "C":[0.07,0.08, 0.09, 0.1, 0.11, 0.12,  1, 10, 100]}
#
#
#
# X, y = make_blobs(n_samples=5000, n_features=3, centers=3, cluster_std=4, random_state=32)
#
# clf = SVC()
#
# gs_model = GridSearchCV(estimator=clf, param_grid=params, cv=5, verbose=2, random_state= 34)
# gs_model.fit(X, y)
#
# print(gs_model.best_params_)
# best_model = gs_model.best_estimator_
#
#



# for kernel in kernels:
#     for c in C :
#         s_t = time.time
#         clf = SVC (kernel=kernel, C=c)
#         clf.fit(X,y)
#         print({f"{kernel:8}, {c:5}, {clf.score(X, y):0.2f}, {time.time()}-{s_t}"})
#



# X, y = make_blobs(n_samples=100, n_features=3, centers=4, random_state=32)
#
#
# clf = SVC(C = 5, kernel="linear", random_state=32)
# clf.fit(X, y)
#
# w = clf.coef_
# b = clf.intercept_
#
# print(w)
# print(b)
#
#
#
# xx, yy = np.meshgrid(
#     np.linspace(X[:, 0].min() -1, X[:, 0].max()
#                 )
# )