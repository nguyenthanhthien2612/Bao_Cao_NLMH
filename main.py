
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_absolute_error
# doc data
def read_data():
    data = pd.read_csv("E:\CT202_NLMH\Bai_Tap_Lon\diabetes.csv")
    # phan X_train, X_test, Y_train, Y_test
    X_train, X_test, Y_train, Y_test = train_test_split(data.iloc[:,0:8], data.Outcome, test_size=0.2, random_state=50)
    return data,X_train, X_test, Y_train, Y_test
# train model dua vao chi so gini
def My_tree_gini(X_train, X_test, Y_train, Y_test):
    clf_gini =DecisionTreeClassifier(criterion="gini", random_state=50)
    clf_gini.fit(X_train,Y_train)
    Y_pred = clf_gini.predict(X_test)
    print("Y_test: ")
    print(Y_pred[0:5])
    print("Y_pred: ")
    print(Y_test[0:5])
    
    print("Do chinh xac cho gia tri du doan: ")
    print("Accuraty is: ", accuracy_score(Y_test, Y_pred)*100)
    # print(confusion_matrix(Y_test,Y_pred, labels=[0,1]))
# train model dua  tren do loi thong tin
def My_tree_entropy(X_train, X_test, Y_train, Y_test):
    clf_entropy = DecisionTreeClassifier(criterion="entropy", random_state=50)
    clf_entropy.fit(X_train,Y_train)
    Y_pred = clf_entropy.predict(X_test)
    print("Y_test: ")
    print(Y_pred[0:5])
    print("Y_pred: ")
    print(Y_test[0:5])

    print("Do chinh xac cho gia tri du doan: ")
    # print(confusion_matrix(Y_test,Y_pred, labels=[0,1]))
    print("Accuraty is: ", accuracy_score(Y_test, Y_pred)*100)


def main():
    data,X_train, X_test, Y_train, Y_test = read_data()
    print("Outcome: ")
    print(data['Outcome'].value_counts())
    print("=====================================================================")
    My_tree_gini(X_train, X_test, Y_train, Y_test)
    print("=====================================================================")
    My_tree_entropy(X_train, X_test, Y_train, Y_test)    




if __name__ =="__main__":
    main()
