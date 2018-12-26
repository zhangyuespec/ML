from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression, SGDRegressor, Ridge  # 导入正规方程和梯度下降
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
import numpy as np
from sklearn.metrics import mean_squared_error


def mylinear():
    """
    线性回归预测房子价格
    :return: None
    """
    # 获取数据
    Boston = load_boston()
    print(Boston)
    print(Boston.target)

    # 分割数据到训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(Boston.data, Boston.target, test_size=0.25)

    # 对特征值标准化处理，和分类算法不一样，这里的目标值也需要标准化处理，目标值最后要reverse_transform()
    # 要实例化两个标准化api
    std_x = StandardScaler()
    x_train = std_x.fit_transform(x_train)
    x_test = std_x.transform(x_test)

    std_y = StandardScaler()
    y_train = std_y.fit_transform(y_train.reshape(-1, 1))
    y_test = std_y.transform(y_test.reshape(-1, 1))  # 0.19版本的转换器，要求传进去的参数必须是二维的

    # 正规方程求解数据结果
    lr = LinearRegression()
    lr.fit(x_train, y_train)
    #保存训练好的模型
    joblib.dump(lr, 'lr.pkl')


if __name__ == '__main__':
    mylinear()
