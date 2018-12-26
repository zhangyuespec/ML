from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
import pandas as pd

li = load_iris()
print(li.data)
print(li.target)
print(li.DESCR)
print(li.target_names)

# 取出数据中的特征值和目标值
x = li.data
y = li.target

# 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

# 对训练集和测试集的特征值标准化
std = StandardScaler()
x_train = std.fit_transform(x_train)
x_test = std.transform(x_test)

# k—近邻算法
knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(x_train, y_train)

predict = knn.predict(x_test)
print("预测目标的种类是", predict)

print("预测目标的准确率是", knn.score(x_test, y_test))
