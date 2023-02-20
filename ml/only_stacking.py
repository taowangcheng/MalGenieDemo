# coding: utf8

import jsonlines
import numpy as np


from sklearn.naive_bayes import MultinomialNB, BernoulliNB, ComplementNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


import pickle
# 二分类
default_classify_modes = {
    'mode1': "two categories",
}

# 两种数学表示方法
default_represent_modes = {
    'mode2': "tfidf",
}

# 设置默认路径
default_input_train_files = {'name': '../data/train_test/train_raw/name.jsonl',
                             'topics': '../data/train_test/train_raw/topics.jsonl',
                             'files': '../data/train_test/train_raw/files.jsonl',
                             'readme': '../data/train_test/train_raw/readme.jsonl',
                             'description': '../data/train_test/train_raw/description.jsonl',
                             'label': '../data/train_test/train_raw/label.jsonl'}

default_fields = ['name', 'topics', 'files', 'readme', 'description']

# 选择字段
default_combine_modes = {
    'mode1': "all fields",
}

# 设置输出文件路径
default_input_file = "../data/train_test/train_test.npz"

# 实例化机器学习方法
clf_all = [
    StackingClassifier(estimators=[
        ('mnb', MultinomialNB()), ('gnb', GaussianNB()), ('bnb', BernoulliNB()), ('cnb', ComplementNB()),
        ('rf', RandomForestClassifier()), ('dt', DecisionTreeClassifier()), ('svc', svm.SVC()),
        ('lr', LogisticRegression()), ('knn', KNeighborsClassifier()), ('lda', LinearDiscriminantAnalysis())
    ]),
]

clf_name = [

    'StackingClassifier',
]


# 机器学习数据预整理
def load_features(fields=None, input_files=None, classify_modes=None, represent_modes=None, combine_modes=None):
    if fields is None:
        fields = default_fields
    if input_files is None:
        input_files = default_input_file
    if classify_modes is None:
        classify_modes = default_classify_modes
    if represent_modes is None:
        represent_modes = default_represent_modes
    if combine_modes is None:
        combine_modes = default_combine_modes
    data = np.load(input_files, allow_pickle=True)
    data_dict = data['data_dict'][()]
    train = []

    for class_mode in classify_modes.keys():
        for represent_mode in represent_modes.keys():
            train_target = data_dict[f"{class_mode}-{represent_mode}-train-label"]
            train_features = {}
            for key_tag in fields:
                train_features[f"{key_tag}"] = data_dict[f"{class_mode}-{represent_mode}-train-{key_tag}"]
            for mode in combine_modes.keys():
                if mode == 'mode1':
                    train = np.concatenate((train_features[fields[0]], train_features[fields[1]],
                                            train_features[fields[2]], train_features[fields[3]],
                                            train_features[fields[4]]), axis=1)

                ml(train, train_target)


# 机器学习方法调用函数
def ml(x_train, y_train):
    target_names = {
        'mode1': ['good', 'malware'],
    }
    for clf, name in zip(clf_all, clf_name):
        clf.fit(x_train, y_train)
        pickle.dump(clf, open("../data/model/ml/stacking.pkl", "wb"))

if __name__ == '__main__':
    load_features()
    pass
