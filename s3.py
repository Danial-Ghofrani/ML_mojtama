import numpy as np
#
# # Data Extraction
# X = np.array([[10],
#      [15],
#      [7],
#      [4]])
# y = np.array([11, 12, 15, 16]).reshape(-1,1)
#
#
# # Linear regression
# sample, feature = X.shape
#
#
#
#
# o = np.ones((feature, 1))
#
#
# theta = [0,0]
#
# X_bias = np.c_[o, X]
# sample, feature = X_bias.shape
#
#
# print(X_bias)
#
# theta = np.zeros((feature,1))
#
# print(X_bias) # 4*2
# print(theta) # 2*1
#
# # Hyper parameters
# learning_rate = 0.01
# max_iter = 100
# cost_function = []
# t = 0.0001
# repeat = 4
#
#
# for i in range(max_iter):
#      # Hypothesis (predict)
#
#      h = np.dot(X_bias, theta)
#      error = np.sqrt(np.mean(np.power(h - y, 2)))
#
#      # gradient Descent
#      gradient = np.dot(X_bias.T, error)/ sample
#
#      # Updating Weights and bias
#      theta = theta - learning_rate * gradient
#
#      rmse = np.sqrt(np.mean(np.power(error, 2)))
#      cost_function.append(rmse)
#
#
#      # earnly stopping
#      if np.array(cost_function[-4:]) < t:
#           break




# ### part 2
#
# # Data Extraction
# X = np.array([[10],
#      [15],
#      [7],
#      [4]])
# y = np.array([11, 12, 15, 16]).reshape(-1,1)
#
#
# # Linear regression
# sample, feature = X.shape
#
#
#
#
# o = np.ones((feature, 1))
#
#
# theta = [0,0]
#
# X_bias = np.c_[o, X]
# sample, feature = X_bias.shape
#
#
# print(X_bias)
#
# theta = np.zeros((feature,1))
#
# print(X_bias) # 4*2
# print(theta) # 2*1
#
# # Hyper parameters
# learning_rate = 0.01
# max_iter = 100
# cost_function = []
# t = 0.0001
# repeat = 4
#
#
# for i in range(max_iter):
#      # Hypothesis (predict)
#
#      h = np.dot(X_bias, theta)
#      error = np.sqrt(np.mean(np.power(h - y, 2)))
#
#      # Stochastic Gradient Descent
#      for x in np.random.permutation(X_bias):
#           gradient = np.dot(x.T, error)
#
#           # Updating Weights and bias
#           theta = theta - learning_rate * gradient
#
#           rmse = np.sqrt(np.mean(np.power(error, 2)))
#           cost_function.append(rmse)
#
#
#           # earnly stopping
#           if np.array(cost_function[-4:]) < t:
#                break



# #### part three
### logistic regression

# from matplotlib import pyplot as plt
# from sklearn.linear_model import LinearRegression
# X = np.array([2,5,7,9,11,50,55,57,60,61]).reshape(-1,1)
# y = np.array([0,0,0,0,0,1,1,1,1,1])
#
# linear = LinearRegression()
# linear.fit(X,y)
#
# plt.scatter(X, y, c=y)
# predict = linear.predict(X)
# predict[predict <= 0.5] = 0
# predict[predict >= 0.5] = 1
#
# plt.plot(X, predict)
#
# plt.show()



### part four

# from matplotlib import pyplot as plt
#
#
# ### logistic
# def sigmoid(X):
#     return 1/(1 +np.e ** -X)
#
# X = np.arange(-1000, 1001)
#
# plt.plot(X, sigmoid(X))
# plt.show()





### part five
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# df = pd.read_csv("diabetes.csv")
# X = df.drop(columns="Outcome")
# y = df["Outcome"]
#
# # print(X.shape)
# # print(y.shape)
#
# x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, stratify=y)
#
# model = LogisticRegression(verbose=1, max_iter=500)
#
# model.fit(x_train, y_train)
#
#
# predict = model.predict(X)
# train_pred = model.predict(x_train)
# test_pred = model.predict(x_test)
#
# print(accuracy_score(y, predict))
# print(accuracy_score(y_train, train_pred))
# print(accuracy_score(y_test, test_pred))
#
#
# # Accuracy
# model.score(x_test, y_test)





### part six

# df = pd.read_csv("diabetes.csv")
# X = df.drop(columns="Outcome")
# y = df["Outcome"]
#
#
#
# x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, stratify=y)
#
# solvers = ["lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"]
#
#
# for solver in solvers:
#
#     model = LogisticRegression(max_iter=5000, solver=solver)
#
#     model.fit(x_train, y_train)
#
#
#     predict = model.predict(X)
#     train_pred = model.predict(x_train)
#     test_pred = model.predict(x_test)
#
#     # print(accuracy_score(y, predict))
#     # print(accuracy_score(y_train, train_pred))
#     print(solver, accuracy_score(y_test, test_pred))
#
#
# # Accuracy
# model.score(x_test, y_test)






### part seven

df = pd.read_csv("diabetes.csv")
X = df.drop(columns="Outcome")
y = df["Outcome"]



x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, stratify=y)


model = LogisticRegression(max_iter=5000)

model.fit(x_train, y_train)


predict = model.predict(X)
train_pred = model.predict(x_train)
test_pred = model.predict(x_test)

# print(accuracy_score(y, predict))
# print(accuracy_score(y_train, train_pred))
print(accuracy_score(y_test, test_pred))



import pickle

with open("model.dat", "wb") as file:
    pickle.dump(model, file)



with open("model.dat", "wb") as file:
    model = pickle.load(file)

print(model.predict([[10, 168, 74, 0, 0, 38, 0.537, 34]]))
print(model.predict_proba([[10, 168, 74, 0, 0, 38, 0.537, 34]]))








