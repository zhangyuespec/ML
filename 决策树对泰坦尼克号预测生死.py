import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz


def decision():
    """
    对泰坦尼克号预测生死
    :return: None
    """
    titan = pd.read_csv("http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt")

    # 处理数据，找出特征值和目标值
    x = titan[['pclass', 'age', 'sex']]
    y = titan['survived']
    print(x)

    # 缺失值处理,用age的平均值取填充age的缺失值
    x["age"].fillna(x["age"].mean(), inplace=True)

    # 分割数据到训练集和数据集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

    # 进行特征工程，特征里面要one—hot编码
    """
    为什么是进行字典抽取而不是文本抽取：
        因为文本抽取是要把文本中不同的字符串作为特征值然后做onr—hot编码，但是这里的txt文件
        已经分好了特征值，但是特征值是字符，所以要one-hot编码的话不能文本抽取，只能是字典抽取
        进行字典抽取的时候，fit_transfrom()里面的参数必须是字典，所以要先将特征值转化成字典格式
    """
    dict = DictVectorizer(sparse=False)  # 先实例化文本特征抽取工具
    x_train = dict.fit_transform(x_train.to_dict(orient="records"))  # 转化
    x_test = dict.transform(x_test.to_dict(orient="records"))
    print(dict.get_feature_names())
    print("11111111111111111111111111111111111111111111111111111111111111111111")
    print(x_train)

    # 用决策树进行预测
    dec = DecisionTreeClassifier(max_depth=40)
    dec.fit(x_train, y_train)
    # 预测准确率
    print("预测结果是",dec.predict(x_test))
    print("预测的准确率为", dec.score(x_test, y_test))

    # 到处决策树的结构
    export_graphviz(dec, out_file="./tree.dot", feature_names=['age', 'pclass=1st', 'pclass=2nd', 'pclass=3rd',
                                                               'sex=female', 'sex=male'])

    return None




if __name__ == '__main__':
    decision()

