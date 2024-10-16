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


x_c = np.arange(len(btc))






x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.05, shuffle=False)

x_c_t = np.arange(len(x_test))


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

lin_train_predict = lin_model.predict(x_train)
lasso_train_predict = lasso_model.predict(x_train)
ridge_train_predict = ridge_model.predict(x_train)


lin_test_predict = lin_model.predict(x_test)
lasso_test_predict = lasso_model.predict(x_test)
ridge_test_predict = ridge_model.predict(x_test)



### evaluating
error = root_mean_squared_error

rmse_linear_train = error(y_train, lin_train_predict)
rmse_linear_test = error(y_test ,lin_test_predict)
rmse_linear_all = error(y, lin_all_predict)

rmse_lasso_train = error(y_train, lasso_train_predict)
rmse_lasso_test = error(y_test, lasso_test_predict)
rmse_lasso_all = error(y, lasso_all_predict)

rmse_ridge_train = error(y_train, ridge_train_predict)
rmse_ridge_test = error(y_test, ridge_test_predict)
rmse_ridge_all = error(y, ridge_all_predict)





print(f"the rmse for the lin model in test data: {rmse_linear_test},"
      f" the rmse for the lasso model in test data: {rmse_lasso_test},"
      f" the rmse for the ridge model in test data: {rmse_ridge_test}")

print(f"the rmse for the lin model in train data: {rmse_linear_train},"
      f" the rmse for the lasso model in train data: {rmse_lasso_train},"
      f" the rmse for the ridge model in train data: {rmse_ridge_train}")

print(f"the rmse for the lin model in all data: {rmse_linear_all},"
      f" the rmse for the lasso model in all data: {rmse_lasso_all},"
      f" the rmse for the ridge model in all data: {rmse_ridge_all}")




### plotting and visualizations
plt.plot(x_c_t, y_test, "g",label = "Real Data")
plt.plot(x_c_t, lin_test_predict, "r", label =  "Linear model")
plt.plot(x_c_t, lasso_test_predict, "b", label = "lasso model")
plt.plot(x_c_t, ridge_test_predict, "orange", label = "ridge model")


plt.legend()
plt.show()

