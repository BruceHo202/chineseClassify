import os
import jieba
from sklearn import metrics
import time
import numpy as np

import pickle
from sklearn import svm
from sklearn.utils import Bunch
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB  # 导入多项式贝叶斯算法
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix, confusion_matrix

articles=list()
test_articles=list()
kind=list()
test_kind = list()

def mySave(savepath, content):
    with open(savepath, "wb") as fp:
        fp.write(content)

def myRead(path):
    with open(path, "rb") as fp:
        content = fp.read()
    return content

# 读取bunch对象
def readbunchobj(path):
    with open(path, "rb") as file_obj:
        bunch = pickle.load(file_obj)
    return bunch



def parse_to_bunch(wordbag_path, parse_path, flag):
    global articles, kind, test_articles, test_kind
    kindlist = os.listdir(parse_path)
    bunch = Bunch(target_name=[], label=[], filenames=[], contents=[])
    bunch.target_name.extend(kindlist)
    k=0
    if flag == 0:
        kind = [0 for i in range(len(kindlist))]
    if flag == 1:
        test_kind = [0 for i in range(len(kindlist))]
    for mydir in kindlist:
        this_path = parse_path + mydir + '/'
        file_list = os.listdir(this_path)
        
        for file_path in file_list:
            fullname = this_path + file_path
            bunch.label.append(mydir)
            bunch.filenames.append(fullname)
            bunch.contents.append(myRead(fullname))
            
            if flag == 0:
                kind[k] += 1
                articles.append(k)
            if flag == 1:
                test_kind[k] += 1
                test_articles.append(k)
        k+=1
            
    with open(wordbag_path, "wb") as file_obj:
        pickle.dump(bunch, file_obj)
    print("构建词典成功")

def vector_space(bunch_path, space_path, train_tfidf_path =  None):
    bunch = readbunchobj(bunch_path)
    tfidfspace = Bunch(target_name=bunch.target_name, label=bunch.label, filenames=bunch.filenames, tdm=[],
                       vocabulary={}, word={})
    
    if train_tfidf_path == None:
        vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5)
        tfidfspace.tdm = vectorizer.fit_transform(bunch.contents)
        tfidfspace.vocabulary = vectorizer.vocabulary_
        tfidfspace.word = vectorizer.get_feature_names()
        
        
    else:
        trainbunch = readbunchobj(train_tfidf_path)
        tfidfspace.vocabulary = trainbunch.vocabulary
        vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                                     vocabulary=trainbunch.vocabulary)
        tfidfspace.tdm = vectorizer.fit_transform(bunch.contents)
        tfidfspace.word = vectorizer.get_feature_names()
    with open(space_path, "wb") as file_obj:
        pickle.dump(tfidfspace, file_obj)
    print("tf-idf计算成功")

def calc_result(actual, predict):
    print('平均正确率:{0:.3f}'.format(metrics.accuracy_score(actual, predict)))
    print('平均召回率:{0:0.3f}'.format(metrics.recall_score(actual, predict, average='weighted')))
    print('f1-score:{0:.3f}'.format(metrics.f1_score(actual, predict, average='weighted')))

def myPredict():
    # 导入训练集
    trainpath = "./mysource/train_wordbag/tfdifspace.dat"
    train_set = readbunchobj(trainpath)

    # 导入测试集
    testpath = "./mysource/test_wordbag/testspace.dat"
    test_set = readbunchobj(testpath)
    train_tfidf = train_set.tdm.toarray()
    word = train_set.vocabulary
    test_word = test_set.vocabulary

    test_tfidf = test_set.tdm.toarray()
    non_zero_place = test_set.tdm.nonzero()
    _index = train_set.tdm.nonzero()
    tmpx = _index[0]
    tmpy = _index[1]
    
    _path = './mysource/train/parse/'
    kindlist = os.listdir(_path)
    print('开始训练')
    time0 = time.process_time()
    tfidf = np.zeros((len(kindlist), len(word)))
    
    for k in range(len(_index[0])):
        i = _index[0][k]
        j = _index[1][k]
        tfidf[articles[i]][j] = tfidf[articles[i]][j] + train_tfidf[i][j]
   
    sum = 0
    nsum = np.zeros(len(kindlist))
    # 统计一个所有tfidf总数和每一类tfidf总数
    
    for i in range(len(kindlist)):
        for j in range(len(word)):
            sum += tfidf[i][j]
            nsum[i] += tfidf[i][j]
    p = np.zeros((len(kindlist), len(word)))
    pc = [0 for i in range(len(kindlist))]
    # pc是一类tfidf占总tfidf比例 p[i][j]是一个词占本类的比例
    print(nsum)
    for i in range(len(kindlist)):
        for j in range(len(word)):
            p[i][j] = (tfidf[i][j] + 0.01) / (nsum[i] + sum)
    
    for i in range(len(kindlist)):
        pc[i] = kind[i] / len(articles)
    
    print('训练完成')
    print(f'训练时间：{time.process_time() - time0}')

    p = np.log(p)
    pc = np.log(pc)

    _cnt = 0
    start = 0
    end = -1
    suc = 0
    suc_n = [0 for i in range(len(kindlist))]
    actual = []
    predict = []
    confusion_matrix = np.zeros((len(kindlist), len(kindlist)))
    err_num = 0
    print('开始测试')
    time1 = time.process_time()
    pred_n = [0 for i in range(len(kindlist))]
    for i in range(len(test_tfidf)):
        max = -100000
        
        for _j in range(start, len(non_zero_place[0])):
            if non_zero_place[0][_j] == i:
                end = end + 1
            else:
                break
        for k in range(len(kindlist)):
            pred = pc[k]
            flag = 0
                
            for j in range(start, end):
                pred = pred + p[k][non_zero_place[1][j]] * test_tfidf[i][non_zero_place[1][j]]
                
            if pred > max:
                flag = k
                
                max = pred
        start = end + 1
        end = start
        actual.append(test_articles[i])
        predict.append(flag)
        pred_n[flag] += 1
        if flag == test_articles[i]:
            _cnt += 1
            suc_n[flag] += 1
            suc += 1
            confusion_matrix[test_articles[i]][test_articles[i]] += 1
        else:
            err_num += 1
            confusion_matrix[test_articles[i]][flag] += 1
    print('测试完成')
    print(f'测试时间：{time.process_time()-time1}')
    print('混淆矩阵：')
    print(confusion_matrix)
    print('每类正确率：')
    for i in range(len(kindlist)):
        print(suc_n[i] / test_kind[i])
    print('每类召回率：')
    for i in range(len(kindlist)):
        print(suc_n[i] / (pred_n[i] + 1))
    calc_result(actual, predict)
    

if __name__ == '__main__':

    # 对训练集建立词典
    wordbag_path = "./mysource/train_wordbag/train_set.dat"  # Bunch存储路径
    parse_path = "./mysource/train/parse/"  # 分词后分类语料库路径
    parse_to_bunch(wordbag_path, parse_path, 0)

    # 对测试集建立词典
    wordbag_path = "./mysource/test_wordbag/test_set.dat"  # Bunch存储路径
    parse_path = "./mysource/test/parse/"  # 分词后分类语料库路径
    parse_to_bunch(wordbag_path, parse_path, 1)

    # 计算tf-idf
    bunch_path = "./mysource/train_wordbag/train_set.dat"
    space_path = "./mysource/train_wordbag/tfdifspace.dat"
    vector_space(bunch_path, space_path)

    bunch_path = "./mysource/test_wordbag/test_set.dat"
    space_path = "./mysource/test_wordbag/testspace.dat"
    train_tfidf_path = "./mysource/train_wordbag/tfdifspace.dat"
    vector_space(bunch_path, space_path, train_tfidf_path)
    # 预测
    myPredict()
