import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer



def rfc():
    titannic = pd.read_csv("./titannic.txt")

    x = titannic[["pclass", "age", "sex"]]
    y = titannic["survived"]

    x["age"].fillna(x["age"].mean(), inplace=True)
    # 特征工程
    dict = DictVectorizer()
    x = dict.fit_transform(x.to_dict(orient="records"))

    # 数据分割
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

    # 建立随机森林实例
    randonf = RandomForestClassifier()

    # 进行网格搜索
    param = {"n_estimators": [120, 200, 300, 500, 800, 1200], "max_depth": [5, 10, 20, 100]}
    gs = GridSearchCV(randonf, param_grid=param, cv=10)

    gs.fit(x_train, y_train)
    print("交叉验证最好的验证结果是", gs.best_score_)
    print("最好的参数模型是", gs.best_estimator_)

    #print(dict)


if __name__ == '__main__':
    rfc()
