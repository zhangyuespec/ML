from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression, SGDRegressor, Ridge  # 导入正规方程和梯度下降
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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
    print("权重参数是", lr.coef_)  # 这是权重参数
    # 预测测试集的房子价格
    y_predict = lr.predict(x_test)
    print("测试集的每个房子的价格预测", std_y.inverse_transform(y_predict))  # 转换会标准化之前的房价
    print("正规方程的均方误差", mean_squared_error(std_y.inverse_transform(y_test), y_predict))

    # 用梯度下降
    sgd = SGDRegressor()
    sgd.fit(x_train, y_train)
    print("梯度下降权重参数是", sgd.coef_)  # 这是权重参数
    # 预测测试集的房子价格
    y_sgd_predict = sgd.predict(x_test)
    print("梯度下降测试集的每个房子的价格预测", std_y.inverse_transform(y_sgd_predict))  # 转换会标准化之前的房价
    print("梯度下降的均方误差", mean_squared_error(std_y.inverse_transform(y_test), y_sgd_predict))

    # 岭回归,岭回归因为带有L2正则化，所以得到的权重系数更加可靠，而且通过demo可以看出来岭回归的均方误差比正规方程和梯度下降都要低
    # 就是因为岭回归解决了过拟合的问题
    rd = Ridge(alpha=1.0)
    rd.fit(x_train, y_train)
    print(rd.coef_)
    y_rd_predict = std_y.inverse_transform(rd.predict(x_test))
    print("岭回归预测房子的价格", y_rd_predict)
    print("岭回归的权重参数是", rd.coef_)
    print("均方误差", mean_squared_error(std_y.inverse_transform(y_test), y_rd_predict))


if __name__ == "__main__":
    mylinear()
