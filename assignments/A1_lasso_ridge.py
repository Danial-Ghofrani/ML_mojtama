import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import train_test_split, KFold


btc = yf.download("BTC-USD")


X = btc[["Open", "High", "Low"]].values
y = btc["Close"].values




x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)


lin_model = LinearRegression()
lasso_model = Lasso()
ridge_model = Ridge()

# training
lin_model.fit(x_train, y_train)
lasso_model.fit(x_train, y_train)
ridge_model.fit(x_train, y_train)

# predicting
lin_predict = lin_model.predict(X)
lasso_predict = lasso_model.predict(X)
ridge_predict = ridge_model.predict(X)


# evaluating
lin_y_pred = lin_model.predict(x_test)
lasso_y_pred = lasso_model.predict(x_test)
ridge_y_pred = ridge_model.predict(x_test)


error = root_mean_squared_error
print(f"the rmse for the lin model: {error(lin_y_pred, y_test)},"
      f" the rmse for the lasso model: {error(lasso_y_pred, y_test)},"
      f" the rmse for the ridge model: {error(ridge_y_pred, y_test)}")



### plotting and visualizations
plt.plot(X, y, "g",label = "Real Data")
plt.plot(X, lin_predict, "r", label =  "Linear model")
plt.plot(X, lasso_predict, "b", label = "lasso model")
plt.plot(X, ridge_predict, "y", label = "ridge model")

plt.show()

