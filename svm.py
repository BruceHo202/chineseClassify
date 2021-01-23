import sys
import os
import jieba
import time

import pickle
from sklearn import svm
from sklearn.utils import Bunch
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB  # 导入多项式贝叶斯算法
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix

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

def parse_to_bunch(wordbag_path, parse_path):
    kindlist = os.listdir(parse_path)
    bunch = Bunch(target_name=[], label=[], filenames=[], contents=[])
    bunch.target_name.extend(kindlist)
    for mydir in kindlist:
        this_path = parse_path + mydir + '/'
        file_list = os.listdir(this_path)
        for file_path in file_list:
            fullname = this_path + file_path
            bunch.label.append(mydir)
            bunch.filenames.append(fullname)
            bunch.contents.append(myRead(fullname))  # 读取文件内容
    with open(wordbag_path, "wb") as file_obj:
        pickle.dump(bunch, file_obj)
    
def vector_space(bunch_path, space_path, train_tfidf_path =  None):
    bunch = readbunchobj(bunch_path)
    tfidfspace = Bunch(target_name=bunch.target_name, label=bunch.label, filenames=bunch.filenames, tdm=[],
                       vocabulary={})
    
    if train_tfidf_path == None:
        vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5)
        tfidfspace.tdm = vectorizer.fit_transform(bunch.contents)
        tfidfspace.vocabulary = vectorizer.vocabulary_
    else:
        trainbunch = readbunchobj(train_tfidf_path)
        tfidfspace.vocabulary = trainbunch.vocabulary
        vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                                     vocabulary=trainbunch.vocabulary)
        tfidfspace.tdm = vectorizer.fit_transform(bunch.contents)
    with open(space_path, "wb") as file_obj:
        pickle.dump(tfidfspace, file_obj)
    
def calc_result(actual, predict):
    print(classification_report(actual, predict))

def myPredict():
    # 导入训练集
    trainpath = "./mysource/train_wordbag/tfdifspace.dat"
    train_set = readbunchobj(trainpath)

    # 导入测试集
    testpath = "./mysource/test_wordbag/testspace.dat"
    test_set = readbunchobj(testpath)
    print('开始训练')
    time0 = time.process_time()
    # 训练分类器
    # 朴素贝叶斯：
    if sys.argv[1] == 'bayes':
        # alpha：平滑参数 fit_prior：是否考虑先验概率，class_prior：是否输入先验概率
         myClassfication = MultinomialNB(alpha=0.001, fit_prior=True)
    # SGD分类：
    elif sys.argv[1] == 'SGD':
        myClassfication = SGDClassifier(loss='hinge', penalty='l2', alpha=0.001)
    # SVM分类：
    elif sys.argv[1] == 'SVM':
        # penalty：惩罚中使用的规范 loss：损失函数 C：错误项惩罚参数 random_state：随机种子 max_iter：最大迭代次数
        myClassfication = svm.LinearSVC(penalty='l2', loss='hinge', C=1.0, random_state=None, max_iter=500)
    else:
        print('参数错误')
        exit(1)
    
    myClassfication.fit(train_set.tdm, train_set.label)
    
    print('训练结束')
    print(f'训练时间：{time.process_time() - time0}')
    print('开始测试')
    
    # 预测分类结果
    predicted = myClassfication.predict(test_set.tdm)

#    for flabel, file_name, expct_cate in zip(test_set.label, test_set.filenames, predicted):
#        if flabel != expct_cate:
#            print(file_name, "  实际类别:", flabel, "   预测类别:", expct_cate)

    print("预测结束")
    time1 = time.process_time()
    print(f'测试时间：{time.process_time() - time1}')
    calc_result(test_set.label, predicted)

if __name__ == '__main__':
    if(len(sys.argv) != 2):
        print(f'输入格式错误，输入示例:\npython try.py bayes\npython try.py SGD\npython try.py SVM')
        exit(1)
    if sys.argv[1] != 'bayes' and sys.argv[1] != 'SGD' and sys.argv[1] != 'SVM':
        print('参数错误')
        exit(1)

    # 对训练集建立词典
    wordbag_path = "./mysource/train_wordbag/train_set.dat"  # Bunch存储路径
    parse_path = "./mysource/train/parse/"  # 分词后分类语料库路径
    parse_to_bunch(wordbag_path, parse_path)
    print("建立训练集词典成功")

    # 对测试集建立词典
    wordbag_path = "./mysource/test_wordbag/test_set.dat"  # Bunch存储路径
    parse_path = "./mysource/test/parse/"  # 分词后分类语料库路径
    parse_to_bunch(wordbag_path, parse_path)
    print("建立测试集词典成功")

    # 计算tf-idf
    bunch_path = "./mysource/train_wordbag/train_set.dat"
    space_path = "./mysource/train_wordbag/tfdifspace.dat"
    vector_space(bunch_path, space_path)
    print("训练集tf-idf计算成功")

    bunch_path = "./mysource/test_wordbag/test_set.dat"
    space_path = "./mysource/test_wordbag/testspace.dat"
    train_tfidf_path = "./mysource/train_wordbag/tfdifspace.dat"
    vector_space(bunch_path, space_path, train_tfidf_path)
    print("测试集tf-idf计算成功")
    # 预测
    myPredict()
