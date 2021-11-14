from numpy.core.defchararray import mod, split
from numpy.core.fromnumeric import product
import pandas as pd 
import numpy as np
from sklearn import model_selection

import  matplotlib.pyplot as plt


data = pd.read_csv('E:\CT202_NLMH\Bai_Tap_Lon\diabetes.csv', delimiter=',')

print(data.shape)
print(data.head(5))

# Câu 1: Dữ liệu có bao nhiêu thuộc tính? Cột nào là cột nhãn? Giá trị của các nhãn (ghi lại kết quả và code vào file nộp bài)

print(list(data.columns[0:-1]))

print(len(data.columns[:-1]))

print('cot nhan la : ', data.columns[-1])

print(np.unique(data.Outcome))
print(data.Outcome.value_counts())



# cau 2: Với tập dữ liệu wineWhite sử dụng nghi thức K-Fold để phân chia tập dữ liệu huấn luyện với
# K=50, sử dụng tham số “Shuffle” để xáo trộn tập dữ liệu trước khi phân chia.
# Xác định số lượng phần tử có trong tập test và tập huấn luyện nếu sử dụng nghi thức đánh giá này



X = data.iloc[ : , 0:-1]
print('Du lieu X: ')
print(X.head(5))

Y = data.iloc[:, -1]
print('Du lieu Y: ')
print(Y.head(5))

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold

kf = KFold(n_splits=10, shuffle= True,random_state= 3000)

list_acc = np.array([0])

for train_index , test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
    print('X train: ')
    print(X_train.head(5))
    print('Y train: ')
    print(Y_train.head(5))

# cau 3: Sử dụng giải thuật Bayes thơ ngây (hàm sklearn) để dự đoán nhãn cho tập kiểm tra theo nghi thức đánh giá của câu 2 với phân phối xác suất Gaussia
    model = GaussianNB()
    model.fit(X_train,Y_train)

# cau 4: Đánh giá độ chính xác cho từng phân lớp dựa vào giá trị dự đoán của câu 3 cho mỗi lần lặp. 
# Chép lại kết quả độ chính xác cho từng phân lớp của lần lặp cuối nộp lại (có thể đưa vào 
# comment trong file code)
    y_pred = model.predict(X_test)
    print('ket qua dat duoc: ')
    print(np.unique(Y_test))
    matrix = confusion_matrix(Y_test,y_pred)
    print(matrix)
    print("-------------------------")
    
# cau 5:  Tính độ chính xác tổng thể cho mỗi lần lặp và độ chính xác tổng thể của trung bình 40 lần lặp 

    print('do chinh xac tong the: ')
    acc = accuracy_score(Y_test,y_pred)
    list_acc = np.append(list_acc, acc)
    print('do chinh xac la :', acc)
    

total = np.sum(list_acc)
print('Do chinh xac tong the cua trung binh 50 lan lap la: ')
print(total/len(list_acc))


# cau 6:  Sử dụng giải thuật Cây quyết định (Decision Tree) ở buổi thực hành số 2 để so sánh hiệu quả
# phân lớp của giải thuật Bayes thơ ngây và cây quyết định với nghi thức đánh giá hold-out


from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
acc_tree = np.array([])
acc_bayes =np.array([])
acc_perceptron = np.array([])

for i in range(10):
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y,random_state=1000+i*5, test_size=1/3.0)
    print('X train: ')
    print(X_train.head(5))
    print('Y train: ')
    print(Y_train.head(5))

    tree_model = DecisionTreeClassifier(criterion="entropy",max_depth=10,random_state=100,min_samples_leaf=9)
    bayes_model = GaussianNB()
    perceptron_model = Perceptron(max_iter=5,eta0=0.02)

    tree_model.fit(X_train,Y_train)
    bayes_model.fit(X_train,Y_train)
    perceptron_model.fit(X_train,Y_train)

    y_pred_tree = tree_model.predict(X_test)
    acc_tree = np.append(acc_tree,accuracy_score(Y_test,y_pred_tree)*100)
    print('Do chinh xac tong the cua Tree dat: ', accuracy_score(Y_test,y_pred_tree)*100 )

    dudoan_bayes= bayes_model.predict(X_test)
    acc_bayes = np.append(acc_bayes,accuracy_score(Y_test,dudoan_bayes)*100)
    print('Do chinh xac tong the cua bayes dat: ', accuracy_score(Y_test,dudoan_bayes)*100)

    y_pred_perceptron = perceptron_model.predict(X_test)
    acc_perceptron = np.append(acc_perceptron,accuracy_score(Y_test,y_pred_perceptron)*100)
    print('Do chinh xac tong the cua Perceptron dat: ', accuracy_score(Y_test,y_pred_tree)*100 )

print("================So Sanh==================")

total_acc_tree = np.sum(acc_tree)
print('Do chinh xac tong the cua trung binh cua giai thuat Tree: ')
print(total_acc_tree/len(acc_tree))


total_acc_bayes = np.sum(acc_bayes)
print('Do chinh xac tong the cua trung binh cua giai thuat Bayes: ')
print(total_acc_bayes/len(acc_bayes))

total_acc_perceptron = np.sum(acc_perceptron)
print('Do chinh xac tong the cua trung binh cua giai thuat Perceptron: ')
print(total_acc_perceptron/len(acc_perceptron))

cot_Y = [1,2,3,4,5,6,7,8,9,10]

plt.plot(cot_Y, acc_tree)
plt.plot(cot_Y,acc_bayes)
plt.plot(cot_Y,acc_perceptron)
plt.show()






