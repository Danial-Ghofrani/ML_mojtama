import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, roc_curve, auc

import pandas as pd
import numpy as np

df = pd.read_csv("final_flood_data.csv")
# There are some Nan rows. We need to replace or drop them, so we don't get any error during fit.
# print(df.isna().any())
# print(df.isna().sum())
# print(df[(df["snow_depth"].isna()) & (df["flood"] == 1 )].shape[0])
# There are 480 missing data and only 47 rows belong to the data with flood=1, so we can drop them with no concern.
df = df.dropna()
# print(df.isna().sum())


scaler = StandardScaler()


X = df.drop(columns="flood")
X = scaler.fit_transform(X)
y = df["flood"]


# print(np.unique(y))



x_train, x_test, y_train, y_test = train_test_split(X, y,test_size=0.2, shuffle=True, stratify=y, random_state=23)



def finding_best_solver():
    '''cheking the best solver by hand and not using param grid.'''
    solvers = ["lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"]


    for solver in solvers:
        LR_model = LogisticRegression(max_iter=3000,solver = solver)
        LR_model.fit(x_train, y_train)

        ## evaluation:
        predict_x_train = LR_model.predict(x_train)
        predict_x_test = LR_model.predict(x_test)
        predict_X = LR_model.predict(X)

        # print(accuracy_score(predict_X, y))
        # print(accuracy_score(predict_x_train, y_train))
        # print(accuracy_score(predict_x_test, y_test))

        print(solver, accuracy_score(predict_x_test, y_test))


def model_roc_curve():
    LR_model = LogisticRegression(max_iter=3000)
    LR_model.fit(x_train, y_train)
    predict_x_test = LR_model.predict(x_test)
    y_test_proba = LR_model.predict_proba(x_test)[:,1]
    fpr, tpr, threshold = roc_curve(y_test, y_test_proba)
    print("Plotting the ROC curve for label y =1 :")
    plt.plot(fpr, tpr, label = "Logistic regression")
    plt.plot([0,1], [0,1], "k--")

    plt.xlim([0,1])
    plt.ylim([0,1])

    plt.xlabel("False Positive Rate(FPR)")
    plt.ylabel("True Positive Rate(TPR)")

    plt.title("Receiver operating characteristics(ROC)")

    model_auc = auc(fpr, tpr)
    print(f"model's auc for y=1 is : {model_auc}")



    y_test_proba = LR_model.predict_proba(x_test)[:, 0]
    fpr, tpr, threshold = roc_curve(y_test, y_test_proba)
    print("Plotting the ROC curve for label y =0 :")
    plt.plot(fpr, tpr, label="Logistic regression")
    plt.plot([0, 1], [0, 1], "k--")

    plt.xlim([0, 1])
    plt.ylim([0, 1])

    plt.xlabel("False Positive Rate(FPR)")
    plt.ylabel("True Positive Rate(TPR)")

    plt.title("Receiver operating characteristics(ROC)")

    model_auc = auc(fpr, tpr)
    print(f"model's auc for y = 0 is : {model_auc}")


    plt.show()




# finding_best_solver()

model_roc_curve()