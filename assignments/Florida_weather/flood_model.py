import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

import pandas as pd
import numpy as np

df = pd.read_csv("final_flood_data.csv")
# There are some Nan rows. We need to replace or drop them, so we don't get any error during fit.
# print(df.isna().any())
# print(df.isna().sum())
# print(df[(df["snow_depth"].isna()) & (df["flood"] == 1 )].shape[0])
# There are 480 missing data and only 47 rows belong to the data with flood=1, so we can drop them with no concern.
df = df.dropna()
print(df.isna().sum())


scaler = StandardScaler()


X = df.drop(columns="flood")
X = scaler.fit_transform(X)
y = df["flood"]


# print(np.unique(y))



x_train, x_test, y_train, y_test = train_test_split(X, y,test_size=0.2, shuffle=True, stratify=y, random_state=23)
LR_model = LogisticRegression(max_iter=100)
LR_model.fit(x_train, y_train)

## evaluation:
predict_x_train = LR_model.predict(x_train)
predict_x_test = LR_model.predict(x_test)
predict_X = LR_model.predict(X)

print(accuracy_score(predict_X, y))
print(accuracy_score(predict_x_test, y_test))
print(accuracy_score(predict_x_train, y_train))