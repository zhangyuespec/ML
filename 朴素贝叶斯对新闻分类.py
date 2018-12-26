from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report


def naive_bayes():
    """
    朴素贝叶斯
    :return: None
    """
    news = fetch_20newsgroups(data_home="./", subset="all")
    # 进行数据分割
    x_train, x_test, y_train, y_test = train_test_split(news.data, news.target, test_size=0.25)

    # 对数据集进行特征抽取
    tf = TfidfVectorizer()

    # 以训练集中的词列表进行每篇文章的重要性统计
    x_train = tf.fit_transform(x_train)

    # print(tf.get_feature_names())

    x_test = tf.transform(x_test)

    # 进行朴素贝叶斯算法的计算
    mlt = MultinomialNB(alpha=1.0)
    # print(x_train.torray())
    mlt.fit(x_train, y_train)  # 训练

    y_predict = mlt.predict(x_test)
    print("预测的类别是", y_predict)

    # 得出准确率
    print("贝叶斯的准确率是", mlt.score(x_test, y_test))

    print("每个类别的精确率和召回率", classification_report(y_test, y_predict, target_names=news.target_names))


if __name__ == '__main__':
    naive_bayes()
