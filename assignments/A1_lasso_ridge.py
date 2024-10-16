import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import train_test_split, KFold


btc = yf.download("BTC-USD")
btc["shifted_close"] = btc["Close"].shift(-1)
btc = btc.dropna()

X = btc[["Open", "Close", "High", "Low"]].values
y = btc["shifted_close"].values







x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
lin_model = LinearRegression()
lasso_model = Lasso()
ridge_model = Ridge()


# training
lin_model.fit(x_train, y_train)
lasso_model.fit(x_train, y_train)
ridge_model.fit(x_train, y_train)


# predicting
lin_all_predict = lin_model.predict(X)
lasso_all_predict = lasso_model.predict(X)
ridge_all_predict = ridge_model.predict(X)

lin_train_predict = lin_model.predict(x_test)
lasso_train_predict = lasso_model.predict(x_test)
ridge_train_predict = ridge_model.predict(x_test)


lin_test_predict = lin_model.predict(x_test)
lasso_test_predict = lasso_model.predict(x_test)
ridge_test_predict = ridge_model.predict(x_test)



### evaluating
error = root_mean_squared_error

rmse_linear_train = error(lin_train_predict, y_train)
rmse_linear_test = error(lin_test_predict, y_test)
rmse_linear_all = error(lin_all_predict, y)

rmse_lasso_train = error(lasso_train_predict, y_train)
rmse_lasso_test = error(lasso_test_predict, y_test)
rmse_lasso_all = error(lasso_all_predict, y)




print(f"the rmse for the lin model: {},"
      f" the rmse for the lasso model: {},"
      f" the rmse for the ridge model: {}")



### plotting and visualizations
plt.plot(X, y, "g",label = "Real Data")
plt.plot(X, lin_predict, "r", label =  "Linear model")
plt.plot(X, lasso_predict, "b", label = "lasso model")
plt.plot(X, ridge_predict, "y", label = "ridge model")

plt.show()

