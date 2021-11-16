
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler
import joblib


def get_metrics(y_test, y_pred_proba):
    print('ACCURACY_SCORE: ', round(accuracy_score(y_test, y_pred_proba >= 0.5), 4))
    print('F1_SCORE: ', round(f1_score(y_test, y_pred_proba >= 0.5), 4))
    print('CONFUSION_MATRIX:\n', confusion_matrix(y_test, y_pred_proba >= 0.5),'\n')


def main():
    data = pd.read_csv('E:\CT202_NLMH\Bai_Tap_Lon\Bao_Cao_Cuoi_Ky\diabetes.csv')
    print("tập dữ liệu: ")
    print(data.head())
    print(data.shape)
    # Tập dữ liệu mà ta sử dụng ở đây có tất cả 9 thuộc tính và 768 hàng dữ liệu

    print("Thống kê tập dữ liệu: ")
    print(data.describe())

    #     Nhìn vào bảng số liệu ta có thể tìm hiểu thêm về các thuộc tính của tập dữ liệu
    # - Glucose: giá trị từ 0 -> 199
    # - BMI: giá trị từ 0 -> 67.1
    # - Insulin: giá trị từ 0 -> 846

    print("kiểm tra giá trị null trong tập dữ liệu: ")
    print(data.isnull().sum())

    # Tập dữ liệu không có bất kì giá trị nào bị thiếu => tập dữ liệu tương đối tốt để huấn luyện mô hình

    # biểu diễn tập dữ liệu Outcome
    zerocount, onecount = data['Outcome'].value_counts()
    totalrows = data.shape[0]
    perOne = (onecount/totalrows)*100
    perZero = (zerocount/totalrows)*100
    print("Tỉ lệ phần trăm của nhãn 1 = "+str(np.round(perOne,2)) + "%")
    print("Tỉ lệ phần trăm của nhãn 0 = "+str(np.round(perZero,2)) + "%")

    labels = 'nhãn 0', 'nhãn 1'
    sizes = [np.round(perZero,2),np.round(perOne,2)]
    explode = (0, 0.1)  # only "explode" the 2nd slice (i.e. 'Hogs')

    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.show()


    sns.countplot(data.Outcome,data=data)
    plt.show()
    # Nhìn vào giá trị dự đoán ta thấy tập dữ nhãn 0 chiếm số lượng nhiều hơn nhãn 1
    # => có thể gây ra tình trạng mô hình sẽ dự đoán không chính xác do mất cân bằng dữ liệu.
    # Chúng ta sẽ giải quyết vấn đề trong phần hướng phát triển của bài toán. 


    corrmat = data.corr()
    fig = plt.figure(figsize = (8,6))

    sns.heatmap(corrmat, vmax = 0.8)
    plt.show()
    # Chúng ta xem mối quan hệ giữa các thuộc tính trong tập dữ liệu

    sns.regplot(x=data['Glucose'], y=data['Age'], )
    plt.show()

    # Nhìn vào biểu đồ Scatter và đường hồi quy ta có nhận xét là tuổi càng cao thì chỉ số Glucose(đường huyết) càng cao 
    # => người càng lớn tuổi thì sẽ nguy cơ bị tiểu đường càng cao

    print("phân tách dữ liệu và nhãn:")
    X = data.iloc[:,0:8]
    Y = data.iloc[:,8]
    print("dữ liệu X: ")
    print(X.iloc[0:5])
    print("nhãn Y: ")
    print(Y.iloc[0:5])

    # phân chia tập dữ liệu
    X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.3, random_state=1)


    # model bayes thơ ngây
    model1 = GaussianNB()
    model1.fit(X_train, y_train)

    y_predict = model1.predict(X_test)
    get_metrics(y_test, y_predict)
    

    # # model cây quyết định
    # model2 = DecisionTreeClassifier()
    # model2.fit(X_train, y_train)

    # y_predict = model2.predict(X_test)
    # get_metrics(y_test, y_predict)

    # # model Perceptron
    # model3 = Perceptron(eta0=0.02,max_iter=5)
    # model3.fit(X_train, y_train)

    # y_predict = model3.predict(X_test)
    # get_metrics(y_test, y_predict)


    # # các model để thử nghiệm
    # model = [GaussianNB(), DecisionTreeClassifier(), Perceptron()] 
    # data_result = {}
    # score = pd.DataFrame(data_result)
    # accuracylist = []

    # for i in range (0,len(model)):
    #     print( model[i].__class__.__name__,".....")
    #     model[i].fit(X_train, y_train)
    #     y_pred = model[i].predict(X_test)
    #     new_row = {'MODEL': model[i].__class__.__name__, 
    #            'ACCURACY_SCORE': round(accuracy_score(y_test, y_pred >= 0.5), 4)}
    #     score = score.append(new_row, ignore_index = True)
    #     accuracylist.append({'Model': model[i].__class__.__name__,'Accuracy': accuracy_score(y_test,y_pred)})

    # score = score.reindex(columns=['MODEL', 'ACCURACY_SCORE'])
    # print(score)

    # plt.figure(figsize=(20, 20))
    # acc_df = pd.DataFrame(accuracylist)
    # acc_df.plot.bar()
    # plt.xticks(np.arange(0, 3),acc_df['Model'],rotation=60,ha = 'right')
    # plt.show()

    # train lần cuối và lưu model
    kf = KFold(n_splits=100, shuffle= True,random_state= 1)
    list_acc = np.array([0])
    max = 0
    for train_index , test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]

        model1.fit(X_train,Y_train)
        y_pred = model1.predict(X_test)

        print('độ chính xác tổng thể: ')
        acc = accuracy_score(Y_test,y_pred)
        list_acc = np.append(list_acc, acc)

        print('accuracy_score :', acc)
        if(acc > max ):
            max = acc
            filename = 'model.pkl'
            joblib.dump(model1, filename)

    total = np.sum(list_acc)
    print('Độ chính xác trung bình của ',len(list_acc),' lần lặp là: ')
    print(total/len(list_acc))

    print('Độ chính xác lớn nhất đạt được là : ', max)    

    
if __name__ == "__main__":
    main()


