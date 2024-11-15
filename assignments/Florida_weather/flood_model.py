import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score

import pandas as pd
import numpy as np

df = pd.read_csv("final_flood_data.csv")



X = df.drop(columns="flood")
y = df["flood"]


# print(np.unique(y))


x_train, x_test, y_train, y_test = train_test_split(X, y,test_size=0.2, shuffle=True, stratify=y, random_state=23)
LR_model = LogisticRegression()
LR_model.fit(x_train, y_train)

## evaluation:
predict_x_train = LR_model.predict(x_train)
predict_x_test = LR_model.predict(x_test)
predict_X = LR_model.predict(X)

print(accuracy_score(predict_X, y))
print(accuracy_score(predict_x_test, y_test))
print(accuracy_score(predict_x_train, y_train))