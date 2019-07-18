from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from gensim.models.word2vec import Word2Vec
import numpy as np
import jieba
import stopWords.stop as stop

stop.init('stopWords/stopWordList(gen).txt')
n_dim = 300

'''
数据预处理（二分类自动生成y版）：
           分词处理
           切分训练集和测试集
'''
def load_file_and_processing(neg,pos):
    def senToWord(sen): # 分词+去停用词
        sou=list(jieba.cut(sen))
        result = []
        for i in sou:
            if not stop.isStopWord(i):
                result.append(i)
        return result

    pos = [senToWord(i) for i in pos]
    neg = [senToWord(i) for i in neg]

    y = np.concatenate((np.ones(len(pos)),np.zeros(len(neg)))) # 1是积极，0是消极
    x = np.concatenate((pos,neg))

    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)

    np.save('y_train.npy', y_train)
    np.save('y_test.npy', y_test)

    return x_train,x_test,y_train,y_test


'''
数据预处理（直接传入xy版）：
           分词处理
           切分训练集和测试集
'''
def load_file_and_processing2(x,y):
    def senToWord(sen): # 分词+去停用词
        sou=list(jieba.cut(sen))
        result = []
        for i in sou:
            if not stop.isStopWord(i):
                result.append(i)
        return result

    x = [senToWord(i) for i in x]

    y = np.array(y)
    x = np.array(x)

    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)

    np.save('y_train.npy', y_train)
    np.save('y_test.npy', y_test)

    return x_train,x_test,y_train,y_test


'''
对每个句子的所有词向量取均值（用于word2Vec）（fix:这个或许可以改进），生成一个句子的vector
'''
def buildSentenceW2v(text, size, model):
    vec = np.zeros(size).reshape((1,size))
    count = 0
    for word in text:
        try:
            vec += model[word].reshape((1,size))
            count += 1
        except KeyError:
            continue
    if count != 0:
        vec /= count
    return vec


'''
计算word2Vec词向量
'''
def getWord2Vec(x_train,x_test):
    # 初始化模型和词表
    model = Word2Vec(size=n_dim,min_count=10)    # 词频少于min_count次数的单词会被丢弃掉, 默认值为5
    model.build_vocab(x_train)

    # 在评论集上训练模型
    model.train(sentences=x_train,total_examples=model.corpus_count,epochs=model.epochs)

    train_vecs = np.concatenate([buildSentenceW2v(z, n_dim, model) for z in x_train])
    np.save('train_vecs.npy',train_vecs)
    print('train_vecs size:')
    print(train_vecs.shape)

    # 在测试集上训练
    model.train(sentences=x_test,total_examples=model.corpus_count,epochs=model.epochs)
    model.save('w2v_model.pkl')
    # build test tweet vector then scale
    test_vecs = np.concatenate([buildSentenceW2v(z, n_dim, model) for z in x_test])
    np.save('test_vecs.npy',test_vecs)
    print('test_vecs size:')
    print(test_vecs.shape)
    return train_vecs, test_vecs, model

stpwrdpath="stopWords/stopWordList(sou).txt"
# 从文件导入停用词表
with open(stpwrdpath, 'rb') as fp:
    stopword = fp.read().decode('gbk')  # 提用词提取
stpwrdlst = stopword.splitlines() # 将停用词表转换为list


'''
计算count词向量
'''
def getCountVec(x_train,x_test):
    x=[' '.join(i) for i in x_train]+[' '.join(i) for i in x_test]
    # vect = CountVectorizer(stop_words=stpwrdlst) # 模型
    vect = CountVectorizer() # 因为一开始加载的时候已经去停用词了，这里可以不再去
    term_matrix = vect.fit_transform(x) # 得到的词向量
    allVec=term_matrix.toarray() # 所有词向量
    model=vect.vocabulary_ # 预览不同词对应的向量维度
    return allVec[:len(x_train)], allVec[len(x_train):], model # 训练集向量，测试集向量，词对应的向量维度
