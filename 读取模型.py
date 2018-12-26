from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression, SGDRegressor, Ridge  # 导入正规方程和梯度下降
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
import numpy as np
from sklearn.metrics import mean_squared_error


def read_lr():
    data = load_boston()
    x_train, x_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.25)

    std = StandardScaler()
    std.fit_transform(y_train)
    std.transform(y_test)

    std_x = StandardScaler()
    std_x.fit_transform(x_train)
    std_x.transform(x_test)
    print(22)
    # 预测结果
    lr = joblib.load('lr.pkl')
    print(std.inverse_transform(lr.predict(x_test)))
    print(111)
