import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso, Ridge


# ##### part one
# ## data cleaning
#
# ## Data Extraction
# btc = yf.download("BTC-USD")
# X = np.arange(len(btc)).reshape(-1, 1)
# y = btc["Close"]
#
# x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, shuffle = False)
#
#
#
#
#
#
# # creating model
# model = LinearRegression()
# # model = np.poly1d(np.polyfit(x_train , y_train , 2))
#
#
#
# #
# slope = model.coef_
# bias = model.intercept_
#
# print(f"y{slope}.X + {bias}")
#
#
# # training data
# model.fit(x_train, y_train)
#
# # evaluating model
# y_pred = model.predict(x_test)
# print(root_mean_squared_error(y_test, y_pred))
#
# # Visualize
# predict = model.predict(X)
#
# plt.plot(X, y, "g", label = "Real Data ")
# plt.plot(X, predict, "b--", label= "Predicted Data")
#
# plt.legend()
# plt.show()















## part two
# btc = yf.download("BTC-USD")
# print(btc.columns)
#
#
# X = btc[["Open", "Low", "High"]]
# y = btc["Close"]
#
# x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, shuffle = False)
#
# # creating model
# model = LinearRegression()
#
#
#
# # training data
# model.fit(x_train, y_train)
#
# #  evaluating model
# y_pred = model.predict(x_test)
# print(root_mean_squared_error(y_test, y_pred))
#
# slope = model.coef_
# bias = model.intercept_
#
# print(f"y{slope}.X + {bias}")
#
#
# # visualization
#
# predict = model.predict(X)
#
# plt.plot(y.values, "g", label = "real data" )
# plt.plot(predict, "b--", label = "prediction")
#
# plt.legend()
# plt.show()







#### part three
btc = yf.download("BTC-USD")
print(btc.columns)


X = np.arange(len(btc))
y = btc["Close"]

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, shuffle = False)

# create and train model
model = np.poly1d(np.polyfit(x_train, y_train, 2))


#  evaluating model
# y_pred = model.predict(x_test)
# print(root_mean_squared_error(y_test, y_pred))




# visualization

predict = model(X)

plt.plot(X, y, "g", label = "real data" )
plt.plot(X, predict, "b--", label = "prediction")

plt.legend()
plt.show()


































## data cleaning
#
# btc = yf.download("BTC-USD")
#
# # X = np.arange(len(btc)).reshape(-1,1)
# X = np.arange(len(btc))
#
# print(X.shape)
# y = btc["Close"]
#
# x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
#
# # creating model
# # model = LinearRegression()
#
# for degree in range(0,10):
#     model = np.poly1d(np.polyfit(x_train, y_train, 2))

# training data
# model.fit(x_train, y_train)
#
# #  evaluating model
# y_pred = model.predict(x_test)
# print(root_mean_squared_error(y_test, y_pred))
#
# slope = model.coef_
# bias = model.intercept_
#
# print(f"y{slope}.X + {bias}")
# model.


# visualization

#     predict = model(X)
#     plt.plot(X, predict, label=f"predicted with degree {degree}")
#
# plt.plot(X, y, "g", label="real data")
# plt.plot(X, predict, "b", label="prediction")
# plt.show()
#
#
#
#
#
#
#
#
# model = Lasso(warm_start = True)
#
# model.fit(x_train, y_train)
# model.fit(x_train, y_train)
#
# model = Ridge()

