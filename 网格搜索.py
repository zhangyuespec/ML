# 网格搜索
from sklearn.datasets import load_iris, fetch_20newsgroups, load_boston
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV


def knncls():
    """
    k近邻预测用户签到位置
    :return: None
    """
    # 读取数据
    data = pd.read_csv('./train.csv')
    print(data.head(10))

    # 处理数据
    # 1.缩小数据的范围，查询数据筛选
    data = data.query("x > 1.0 & x < 1.25 & y > 2.5 & y < 2.75  ")  # query里面字符串填写缩小条件

    # 2.处理时间的数据，时间戳转换成年月日时分秒当作新特征
    time_value = pd.to_datetime(data["time"], unit="s")
    # print(time_value)

    # 3.把日期格式转化成字典格式
    time_value = pd.DatetimeIndex(time_value)

    # 4.构造一些特征，因为年和月是一样的所以不做为特征了
    data["day"] = time_value.day  # 直接增加一个特征
    data["hour"] = time_value.hour
    data["weekday"] = time_value.weekday

    # 5.把原来的时间戳特征删除
    data.drop(["time"], axis=1)
    print(data)

    # 把签到数量少于n的目标位置删除
    place_count = data.groupby("place_id").count()  # data以place_id来分组
    tf = place_count[place_count.row_id > 3].reset_index()  # 分完组后row_id已经变成了place_id的统计和
    # print(tf)
    data = data[data["place_id"].isin(tf.place_id)]  # isin和布尔逻辑结合来清洗数据
    # print(data.head(10))

    # 取出数据中的特征值和目标值
    y = data["place_id"]  # 目标值

    x = data.drop("place_id", axis=1)  # 特征值
    x = x.drop("row_id", axis=1)

    # 分割训练集和测试集
    x_trian, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

    # 特征工程（标准化）
    std = StandardScaler()
    # 对训练集和测试集的特征值标准化
    x_trian = std.fit_transform(x_trian)
    x_test = std.transform(x_test)

    # 进行算法流程
    knn = KNeighborsClassifier()
    # knn.fit(x_trian, y_train)
    #
    # # 得出预测结果
    # y_predict = knn.predict(x_test)
    # print("预测的目标签到位置为：", y_predict)
    #
    # # 得出准确率
    # print("预测的准确率是", knn.score(x_test, y_test))

    # 进行网格搜索
    # 构造一些参数的值用于搜索
    param = {"n_neighbors": [3, 5, 10]}
    gc = GridSearchCV(knn, param_grid =param, cv=10)
    gc.fit(x_trian, y_train)

    # 预测准确率
    print("在测试集上的准确率", gc.score(x_test, y_test))

    print("在交叉验证中最好的结果",gc.best_score_)
    print("选择了最好的模型是",gc.best_estimator_)
    print("每个超参数每个交叉验证的结果",gc.cv_results_)

if __name__ == '__main__':
    knncls()
