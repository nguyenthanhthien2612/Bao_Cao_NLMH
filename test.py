import joblib
from sklearn import model_selection
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd




def main():
    # nhập dữ liệu test

    loaded_model = joblib.load('model.pkl')

    
    while(True):
        loaded_model = joblib.load('model.pkl')
        print("==================nhập dữ liệu dự đoán==================")
        Pregnancies = int(input("Nhập  Pregnancies giá trị từ 0 đến 17: "))
        Glucose = int(input("Nhập  Glucose giá trị từ 0 đến 199: "))
        BloodPressure = int(input("Nhập  BloodPressure giá trị từ 0 đến 122: "))
        SkinThickness = int(input("Nhập SkinThickness giá trị từ 0 đến 99: "))
        Insulin = int(input("Nhập  Insulin giá trị từ 0 đến 846: ")) 
        BMI =  float(input("Nhập  BMI giá trị từ 0.0 đến 67.1: "))
        DiabetesPedigreeFunction = float(input("Nhập  DiabetesPedigreeFunction giá trị từ 0.078 đến 2.42: "))
        Age = int(input("Nhập  Pregnancies giá trị từ 21 đến 81: "))

        X_test = pd.DataFrame([[Pregnancies,  Glucose,  BloodPressure,  SkinThickness,  Insulin,   BMI,  DiabetesPedigreeFunction,  Age]])

        y_pred = loaded_model.predict(X_test)
        print("Kết Quả Train: ")
        print("Kết quả Outcome là: ",y_pred)

        br = int(input("Bạn muốn tiếp tục hay dừng lại!!! Nhập 1 để test, Nhập 0 để dừng : "))
        if (br==1):
            continue
        else:
            break

if __name__ == "__main__":
    main()