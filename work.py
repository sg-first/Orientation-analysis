import numpy as np
from sklearn.naive_bayes import GaussianNB # 这里以高斯贝叶斯为例
from sklearn.externals import joblib    #把数据转化为二进制
from sklearn.svm import SVC
import vec
import jieba
import csv


'''
训练SVM模型
'''
def svm_train(train_vecs,y_train,test_vecs,y_test):
    clf = SVC(kernel='rbf',verbose=True,probability=True)
    clf.fit(train_vecs,y_train)
    joblib.dump(clf, 'model.pkl')
    print('交叉验证得分',clf.score(test_vecs,y_test))
    return clf


'''
构建待测句子的W2v向量
'''
def buildPredictW2v(sen, model):
    allWords = jieba.cut(sen)  # jieba.lcut直接返回list
    train_vecs = vec.buildSentenceW2v(allWords, vec.n_dim, model)
    return train_vecs

'''
构建待测句子的count向量
'''
def buildPredictCountVec(sen,model):
    result = []  # 创建结果向量，model限定维数
    for _ in range(len(model)):
        result.append(0)

    allWords = jieba.cut(sen)  # jieba.lcut直接返回list
    keyList = list(model.keys())
    for word in allWords:
        if word in keyList:
            sub=model[word]
            result[sub]+=1

    return np.array(result)


'''
对单个句子进行情感分析（两个模型都能用）
'''
def predict(words_vecs,clf):
    probability = clf.predict_proba(words_vecs) # 属于各个类的概率
    probability = probability.tolist()[0]
    return probability.index(max(probability)),probability


'''
训练贝叶斯模型
'''
def bayes_train(train_vecs,y_train,test_vecs,y_test):
    clf = GaussianNB()  # 默认priors=None，可用clf.set_params设置各个类标记的先验概率
    clf.fit(train_vecs,y_train)
    joblib.dump(clf, 'model.pkl')
    print('交叉验证得分',clf.score(test_vecs, y_test))
    return clf


if __name__=='__main__':
    link = "./data"  # 路径
    filename = "306537777_20181231_001"  # 答案json文件名
    encoding = "utf-8"
    stpwrdpath = "../stopWords/stopWordList(sou).txt"  # 停用词表路径
    analyCSVpath = "./try1/data/倾向性分析数据集.csv"  # 已评分数据集(CSV文件)路径


    def classi(score):  # 根据评分分成四类
        score = float(score)
        if (score >= 0 and score < 0.25):
            return 0
        if (score >= 0.25 and score < 0.5):
            return 1
        if (score >= 0.5 and score < 0.75):
            return 2
        if (score >= 0.75 and score < 1):
            return 3


    fp2 = open(analyCSVpath, 'r', encoding=encoding)
    analyCSV = csv.reader(fp2)
    X = []  # 答案内容
    y = []  # 对应评分
    for i in analyCSV:
        X.append(i[1])
        y.append(classi(i[2]))

    x_train, x_test, y_train, y_test = vec.load_file_and_processing2(X,y)
    train_vecs, test_vecs, model = vec.getWord2Vec(x_train, x_test)
    clf = svm_train(train_vecs, y_train, test_vecs, y_test)
    words_vecs = buildPredictW2v('我要好好学习',model)
    result = predict(words_vecs,clf)
    print(result)
