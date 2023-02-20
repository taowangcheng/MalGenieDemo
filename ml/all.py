import jsonlines
import numpy as np

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.naive_bayes import MultinomialNB, BernoulliNB, ComplementNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, StackingClassifier, VotingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

result_dict = {}

# 二分类
default_classify_modes = {
    'mode1': "two categories",
}

# 设置数学表示方法
default_represent_modes = {
    'mode1': "count",
    'mode2': "tfidf",
    'mode3': 'word2vec based unweighted average',
    'mode4': 'word2vec based weighted average without pca',
    'mode5': 'word2vec based weighted average with pca',
    'mode6': 'fast_text',
}

# 设置默认路径
default_input_train_files = {'name': '../data/train_test/train_raw/name.jsonl',
                             'topics': '../data/train_test/train_raw/topics.jsonl',
                             'files': '../data/train_test/train_raw/files.jsonl',
                             'readme': '../data/train_test/train_raw/readme.jsonl',
                             'description': '../data/train_test/train_raw/description.jsonl',
                             'label': '../data/train_test/train_raw/label.jsonl'}
default_input_test_files = {'name': '../data/train_test/test_raw/name.jsonl',
                            'topics': '../data/train_test/test_raw/topics.jsonl',
                            'files': '../data/train_test/test_raw/files.jsonl',
                            'readme': '../data/train_test/test_raw/readme.jsonl',
                            'description': '../data/train_test/test_raw/description.jsonl',
                            'label': '../data/train_test/test_raw/label.jsonl'}
default_input_real_files = {'name': '../data/train_test/real_raw/name.jsonl',
                            'topics': '../data/train_test/real_raw/topics.jsonl',
                            'files': '../data/train_test/real_raw/files.jsonl',
                            'readme': '../data/train_test/real_raw/readme.jsonl',
                            'description': '../data/train_test/real_raw/description.jsonl',
                            'label': '../data/train_test/real_raw/label.jsonl'}
default_fields = ['name', 'topics', 'files', 'readme', 'description']

# 选择字段
default_combine_modes = {
    'mode1': "all fields",
    # 'mode2': "all fields except name",
    # 'mode3': "all fields except name and files",
    # 'mode4': "all fields except name and files and readme",
    # 'mode5': "only topics",
    # 'mode6': "only description",
}

# 设置输出文件路径
default_input_file = "../data/train_test/train_test.npz"

# 实例化机器学习方法
clf_all = [
    MultinomialNB(),
    GaussianNB(),
    BernoulliNB(),
    ComplementNB(),
    RandomForestClassifier(random_state=0,n_jobs=10),
    DecisionTreeClassifier(),
    svm.SVC(),
    KNeighborsClassifier(n_jobs=10),
    LogisticRegression(n_jobs=10),
    LinearDiscriminantAnalysis(),

    BaggingClassifier(MultinomialNB(),n_jobs=10),
    BaggingClassifier(GaussianNB(),n_jobs=10),
    BaggingClassifier(BernoulliNB(),n_jobs=10),
    BaggingClassifier(ComplementNB(),n_jobs=10),
    BaggingClassifier(RandomForestClassifier(),n_jobs=10),
    BaggingClassifier(DecisionTreeClassifier(),n_jobs=10),
    BaggingClassifier(svm.SVC(),n_jobs=10),
    BaggingClassifier(KNeighborsClassifier(),n_jobs=10),
    BaggingClassifier(LogisticRegression(),n_jobs=10),
    BaggingClassifier(LinearDiscriminantAnalysis(),n_jobs=10),
    GradientBoostingClassifier(),
    AdaBoostClassifier(),
    StackingClassifier(estimators=[
        ('mnb', MultinomialNB()), ('gnb', GaussianNB()), ('bnb', BernoulliNB()), ('cnb', ComplementNB()),
        ('rf', RandomForestClassifier()), ('dt', DecisionTreeClassifier()), ('svc', svm.SVC()),
        ('lr', LogisticRegression()), ('knn', KNeighborsClassifier()), ('lda', LinearDiscriminantAnalysis())
    ],n_jobs=10),
    VotingClassifier(estimators=[
        ('mnb', MultinomialNB()), ('gnb', GaussianNB()), ('bnb', BernoulliNB()), ('cnb', ComplementNB()),
        ('rf', RandomForestClassifier()), ('dt', DecisionTreeClassifier()), ('svc', svm.SVC(probability=True)),
        ('lr', LogisticRegression()), ('knn', KNeighborsClassifier()), ('lda', LinearDiscriminantAnalysis())
    ], voting='soft', weights=[1, 1, 1, 1, 5, 0, 1, 0, 1, 1],n_jobs=10),
]


clf_name = [
    'MultinomialNB Naive Bayes', 'GaussianNB', 'BernoulliNB', 'ComplementNB', 'Random forest', 'Decision Tree',
    'SVC', 'KNN', 'Logistic Regression', 'Linear Discriminant Analysis',
    'Bagging Classifier mnb', 'Bagging Classifier gnb', 'Bagging Classifier bnb', 'Bagging Classifier cnb',
    'Bagging Classifier rf', 'Bagging Classifier dt', 'Bagging Classifier svm', 'Bagging Classifier knn',
    'Bagging Classifier lr', 'Bagging Classifier lda', 'GradientBoostingClassifier', 'AdaBoostClassifier',
    'StackingClassifier', 'Vote'
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
    test = []
    real = []
    for class_mode in classify_modes.keys():
        for represent_mode in represent_modes.keys():
            train_target = data_dict[f"{class_mode}-{represent_mode}-train-label"]
            test_target = data_dict[f"{class_mode}-{represent_mode}-test-label"]
            real_target = data_dict[f"{class_mode}-{represent_mode}-real-label"]
            train_features = {}
            test_features = {}
            real_features = {}
            for key_tag in fields:
                train_features[f"{key_tag}"] = data_dict[f"{class_mode}-{represent_mode}-train-{key_tag}"]
                test_features[f"{key_tag}"] = data_dict[f"{class_mode}-{represent_mode}-test-{key_tag}"]
                real_features[f"{key_tag}"] = data_dict[f"{class_mode}-{represent_mode}-real-{key_tag}"]
            for mode in combine_modes.keys():
                if mode == 'mode1':
                    train = np.concatenate((train_features[fields[0]], train_features[fields[1]],
                                            train_features[fields[2]], train_features[fields[3]],
                                            train_features[fields[4]]), axis=1)
                    test = np.concatenate((test_features[fields[0]], test_features[fields[1]],
                                           test_features[fields[2]], test_features[fields[3]],
                                           test_features[fields[4]]), axis=1)
                    real = np.concatenate((real_features[fields[0]], real_features[fields[1]],
                                           real_features[fields[2]], real_features[fields[3]],
                                           real_features[fields[4]]), axis=1)
                elif mode == 'mode2':
                    train = np.concatenate((train_features[fields[1]], train_features[fields[2]],
                                            train_features[fields[3]], train_features[fields[4]]), axis=1)
                    test = np.concatenate((test_features[fields[1]], test_features[fields[2]],
                                           test_features[fields[3]], test_features[fields[4]]), axis=1)
                    real = np.concatenate((real_features[fields[1]], real_features[fields[2]],
                                           real_features[fields[3]], real_features[fields[4]]), axis=1)
                elif mode == 'mode3':
                    train = np.concatenate((train_features[fields[1]], train_features[fields[3]],
                                            train_features[fields[4]]), axis=1)
                    test = np.concatenate((test_features[fields[1]], test_features[fields[3]],
                                           test_features[fields[4]]), axis=1)
                    real = np.concatenate((real_features[fields[1]], real_features[fields[3]],
                                           real_features[fields[4]]), axis=1)
                elif mode == 'mode4':
                    train = np.concatenate((train_features[fields[1]], train_features[fields[4]]), axis=1)
                    test = np.concatenate((test_features[fields[1]], test_features[fields[4]]), axis=1)
                    real = np.concatenate((real_features[fields[1]], real_features[fields[4]]), axis=1)
                elif mode == 'mode5':
                    train = train_features[fields[1]]
                    test = test_features[fields[1]]
                    real = real_features[fields[1]]
                elif mode == 'mode6':
                    train = train_features[fields[4]]
                    test = test_features[fields[4]]
                    real = real_features[fields[4]]


                ml(train, train_target, test, test_target, real, real_target, class_mode, represent_mode, mode)

# 机器学习方法调用函数
def ml(x_train, y_train, x_test, y_test, x_real, y_real, class_mode, represent_mode, mode):
    target_names = {
        'mode1': ['good', 'malware'],
    }
    scaler1 = StandardScaler()
    scaler2 = MinMaxScaler()
    if represent_mode != 'mode1' or represent_mode != 'mode2':
        x_train = scaler1.fit_transform(x_train)
        x_test = scaler1.transform(x_test)
        x_real = scaler1.transform(x_real)

        x_train = scaler2.fit_transform(x_train)
        x_test = scaler2.transform(x_test)
        x_real = scaler2.transform(x_real)

    print("_" * 80)
    print(default_classify_modes[class_mode])
    print(default_represent_modes[represent_mode])
    all_classification(x_train, y_train, x_test, y_test, x_real, y_real, target_names[mode]
                       , class_mode, represent_mode)
    np.savez('../data/result_all.npz', mal_dict=result_dict)

# 机器学习方法函数
def benchmark(clf, name, x_train, x_test, x_real, y_train, y_test, y_real, target,
              class_mode, represent_mode):
    print("Training: ")
    print(clf)
    clf.fit(x_train, y_train)
    print('The accuracy of classifying malware in test set', name, clf.score(x_test, y_test))
    pred = clf.predict(x_test)
    score = accuracy_score(y_test, pred)
    print("classification report:")
    print(classification_report(y_test, pred, target_names=target))
    print()

    real_pred = clf.predict(x_real)
    index_list = []
    for index, x in enumerate(real_pred):
        if x == 1:
            index_list.append(index)
    result_dict[f"{class_mode}-{represent_mode}-{name}"] = index_list
    clf_descr = str(clf).split("(")[0]
    return clf_descr, score

# 输出训练效果
def all_classification(x_train, y_train, x_test, y_test, x_real, y_real, target, class_mode, represent_mode):
    results = []
    for clf, name in zip(clf_all, clf_name):
        print("=" * 80)
        print(name)
        results.append(benchmark(clf, name, x_train, x_test, x_real, y_train, y_test, y_real, target,
                                 class_mode, represent_mode))
    print(results)


if __name__ == '__main__':
    load_features()
    pass
