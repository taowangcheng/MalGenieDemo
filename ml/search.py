import jsonlines
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

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


default_classify_modes = {
    'mode1': "two categories",
}

# 数学表示方法
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
    RandomForestClassifier(random_state=0),
    DecisionTreeClassifier(),
    svm.SVC(),
    KNeighborsClassifier(),
    LogisticRegression(),
    LinearDiscriminantAnalysis(),

    BaggingClassifier(MultinomialNB()),
    BaggingClassifier(GaussianNB()),
    BaggingClassifier(BernoulliNB()),
    BaggingClassifier(ComplementNB()),
    BaggingClassifier(RandomForestClassifier()),
    BaggingClassifier(DecisionTreeClassifier()),
    BaggingClassifier(svm.SVC()),
    BaggingClassifier(KNeighborsClassifier()),
    BaggingClassifier(LogisticRegression()),
    BaggingClassifier(LinearDiscriminantAnalysis()),
    GradientBoostingClassifier(),
    AdaBoostClassifier(),
    StackingClassifier(estimators=[
        ('mnb', MultinomialNB()), ('gnb', GaussianNB()), ('bnb', BernoulliNB()), ('cnb', ComplementNB()),
        ('rf', RandomForestClassifier()), ('dt', DecisionTreeClassifier()), ('svc', svm.SVC()),
        ('lr', LogisticRegression()), ('knn', KNeighborsClassifier()), ('lda', LinearDiscriminantAnalysis())
    ]),
    VotingClassifier(estimators=[
        ('mnb', MultinomialNB()), ('gnb', GaussianNB()), ('bnb', BernoulliNB()), ('cnb', ComplementNB()),
        ('rf', RandomForestClassifier()), ('dt', DecisionTreeClassifier()), ('svc', svm.SVC(probability=True)),
        ('lr', LogisticRegression()), ('knn', KNeighborsClassifier()), ('lda', LinearDiscriminantAnalysis())
    ], voting='soft', weights=[1, 1, 1, 1, 5, 0, 1, 0, 1, 1]),
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

    for class_mode in classify_modes.keys():
        for represent_mode in represent_modes.keys():
            train_target = data_dict[f"{class_mode}-{represent_mode}-train-label"]
            test_target = data_dict[f"{class_mode}-{represent_mode}-test-label"]
            train_features = {}
            test_features = {}
            for key_tag in fields:
                train_features[f"{key_tag}"] = data_dict[f"{class_mode}-{represent_mode}-train-{key_tag}"]
                test_features[f"{key_tag}"] = data_dict[f"{class_mode}-{represent_mode}-test-{key_tag}"]
            for mode in combine_modes.keys():
                if mode == 'mode1':
                    train = np.concatenate((train_features[fields[0]], train_features[fields[1]],
                                            train_features[fields[2]], train_features[fields[3]],
                                            train_features[fields[4]]), axis=1)
                    test = np.concatenate((test_features[fields[0]], test_features[fields[1]],
                                           test_features[fields[2]], test_features[fields[3]],
                                           test_features[fields[4]]), axis=1)
                elif mode == 'mode2':
                    train = np.concatenate((train_features[fields[1]], train_features[fields[2]],
                                            train_features[fields[3]], train_features[fields[4]]), axis=1)
                    test = np.concatenate((test_features[fields[1]], test_features[fields[2]],
                                           test_features[fields[3]], test_features[fields[4]]), axis=1)
                elif mode == 'mode3':
                    train = np.concatenate((train_features[fields[1]], train_features[fields[3]],
                                            train_features[fields[4]]), axis=1)
                    test = np.concatenate((test_features[fields[1]], test_features[fields[3]],
                                           test_features[fields[4]]), axis=1)

                elif mode == 'mode4':
                    train = np.concatenate((train_features[fields[1]], train_features[fields[4]]), axis=1)
                    test = np.concatenate((test_features[fields[1]], test_features[fields[4]]), axis=1)

                elif mode == 'mode5':
                    train = train_features[fields[1]]
                    test = test_features[fields[1]]

                elif mode == 'mode6':
                    train = train_features[fields[4]]
                    test = test_features[fields[4]]

                ml(train, train_target, test, test_target, class_mode, represent_mode, mode)

# 机器学习模型超参数搜索方法调用函数
def ml(x_train, y_train, x_test, y_test, class_mode, represent_mode, mode):
    target_names = {
        'mode1': ['good', 'malware'],
    }
    scaler1 = StandardScaler()
    scaler2 = MinMaxScaler()
    if represent_mode != 'mode1' or represent_mode != 'mode2':
        x_train = scaler1.fit_transform(x_train)
        x_test = scaler1.transform(x_test)

        x_train = scaler2.fit_transform(x_train)
        x_test = scaler2.transform(x_test)

    print("_" * 80)
    print(default_classify_modes[class_mode])
    print(default_represent_modes[represent_mode])

    gridsearch(x_train, y_train, x_test, y_test, target_names[mode],
               class_mode, represent_mode)

# 机器学习模型超参数搜索函数
def gridsearch(x_train, y_train, x_test, y_test, target, class_mode, represent_mode):
    clf_mnb = Pipeline([('nb', MultinomialNB())])
    parameters_mnb = {
        'nb__alpha': np.arange(0.1, 4, 0.01),
    }

    clf_gnb = Pipeline([('gnb', GaussianNB())])
    parameters_gnb = {
        # 'gnb__alpha': np.arange(1.0e-10, 3, 0.05),
    }

    clf_bnb = Pipeline([('bnb', BernoulliNB())])
    parameters_bnb = {
        'bnb__alpha': np.arange(1.0e-10, 3, 0.01),
    }

    clf_cnb = Pipeline([('cnb', ComplementNB())])
    parameters_cnb = {
        'cnb__alpha': np.arange(1.0e-10, 3, 0.01),
    }

    clf_rf = Pipeline([('RF', RandomForestClassifier(random_state=0)), ])
    parameters_rf = {
        'RF__n_estimators': np.arange(1, 100, 1),  # 89
        'RF__max_depth': np.arange(1, 80, 1),  # 47

        'RF__min_samples_split': np.arange(2, 10, 1),  # 2         #这两个要一起调
        'RF__min_samples_leaf': np.arange(1, 10, 1),  # 1

        'RF__max_features': np.arange(1, 400, 1),  # 18
    }

    clf_dt = Pipeline([('DTree', DecisionTreeClassifier(random_state=0))])
    parameters_dt = {
        'DTree__criterion': ["gini", "entropy"],  # gini
        'DTree__max_depth': np.arange(2, 10, 1),
        'DTree__min_samples_split': np.arange(2, 20, 1),
        'DTree__min_samples_leaf': np.arange(1, 5, 1),
        'DTree__min_impurity_decrease': np.arange(0.1, 1, 0.1),
    }

    clf_svm = Pipeline([('svm', svm.SVC())])
    parameters_svm = [
        {},
        {'svm__kernel': ['linear'], 'svm__C': [1, 10, 100, 1000]},
        {'svm__kernel': ['poly'], 'svm__C': [1, 10, 100, 1000], 'svm__degree': [1, 2, 3], },
        {'svm__kernel': ['rbf'], 'svm__C': [1, 10, 100, 1000], 'svm__gamma': [1, 0.1, 0.01], }
    ]

    clf_knn = Pipeline([('KNN', KNeighborsClassifier(n_neighbors=11, p=3, weights='distance'))])
    parameters_knn = [
        {},
        {'KNN__weights': ['distance'], 'KNN__n_neighbors': np.arange(1, 12, 1), 'KNN__p': np.arange(1, 8, 1), },
        {'KNN__weights': ['uniform'], 'KNN__n_neighbors': np.arange(1, 12, 1), },
    ]

    clf_lr = Pipeline([('LR', LogisticRegression(C=0.1, max_iter=1000, penalty='l2', solver='sag'))])
    parameters_lr = [
        {},
        {
            'LR__penalty': ['none'],
            'LR__solver': ['sag', 'lbfgs', 'newton-cg', 'saga'],
            # 'LR__C': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.01, 0.1, 1, 10, 100, 1000],
            'LR__max_iter': [1000, 10000, 100000, 1000000, 10000000],
        },
        {
            'LR__penalty': ['l1'],
            'LR__solver': ['liblinear', 'saga'],
            'LR__C': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.01, 0.1, 1, 10, 100, 1000],
            'LR__max_iter': [1000, 10000, 100000, 1000000, 10000000],
        },
        {
            'LR__penalty': ['l2'],
            'LR__solver': ['liblinear', 'sag', 'lbfgs', 'newton-cg', 'saga'],
            'LR__C': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.01, 0.1, 1, 10, 100, 1000],
            'LR__max_iter': [1000, 10000, 100000, 1000000, 10000000],
        },
        {
            'LR__penalty': ['elasticnet'],
            'LR__C': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.01, 0.1, 1, 10, 100, 1000],
            'LR__max_iter': [1000, 10000, 100000, 1000000, 10000000],
        },
    ]

    clf_lda = Pipeline([('LDA', LinearDiscriminantAnalysis())])
    parameters_lda = [
        {},
        {'LDA__solver': ['svd']},
        {'LDA__solver': ['lsqr']},
        {'LDA__solver': ['eigen']}
    ]

    clf_mnb_bag = BaggingClassifier(MultinomialNB())
    parameters_mnb_bag = {
        'base_estimator__alpha': np.arange(1.0e-10, 1, 0.01),
        'max_samples': np.arange(0.1, 0.9, 0.1), 'max_features': np.arange(0.1, 0.9, 0.1)
    }

    clf_gnb_bag = BaggingClassifier(GaussianNB())
    parameters_gnb_bag = {
        'base_estimator__var_smoothing': np.arange(1e-10, 1e-9, 1e-10),
        'max_samples': np.arange(0.1, 0.9, 0.1), 'max_features': np.arange(0.1, 0.9, 0.1)
    }

    clf_bnb_bag = BaggingClassifier(BernoulliNB())
    parameters_bnb_bag = {
        'base_estimator__alpha': np.arange(1.0e-10, 1, 0.01),
        'max_samples': np.arange(0.1, 0.9, 0.1), 'max_features': np.arange(0.1, 0.9, 0.1)
    }

    clf_cnb_bag = BaggingClassifier(ComplementNB())
    parameters_cnb_bag = {
        'base_estimator__alpha': np.arange(1.0e-10, 1, 0.01),
        'max_samples': np.arange(0.1, 0.9, 0.1), 'max_features': np.arange(0.1, 0.9, 0.1)
    }

    clf_rf_bag = BaggingClassifier(RandomForestClassifier())
    parameters_rf_bag = {
        'base_estimator__n_estimators': np.arange(75, 100, 1),
        'base_estimator__max_depth': np.arange(20, 80, 1),

        'base_estimator__min_samples_split': np.arange(2, 10, 1),  # 这两个要一起调
        'base_estimator__min_samples_leaf': np.arange(1, 10, 1),

        'base_estimator__max_features': np.arange(1, 20, 1),
        'max_samples': np.arange(0.1, 0.9, 0.1), 'max_features': np.arange(0.1, 0.9, 0.1)
    }

    clf_dt_bag = BaggingClassifier(DecisionTreeClassifier())
    parameters_dt_bag = {
        'base_estimator__criterion': ["gini", "entropy"],
        'base_estimator__max_depth': np.arange(1, 10, 1),
        'base_estimator__min_samples_split': np.arange(2, 20, 1),
        'base_estimator__min_samples_leaf': np.arange(1, 5, 1),
        'base_estimator__min_impurity_decrease': np.arange(0.1, 1, 0.1),
        'max_samples': np.arange(0.1, 0.9, 0.1), 'max_features': np.arange(0.1, 0.9, 0.1)
    }

    clf_svm_bag = BaggingClassifier(svm.SVC())
    parameters_svm_bag = [
        {'base_estimator__kernel': ['linear'], 'base_estimator__C': [1, 10, 100, 1000],
         'max_samples': np.arange(0.1, 0.9, 0.1), 'max_features': np.arange(0.1, 0.9, 0.1)},
        {'base_estimator__kernel': ['poly'], 'base_estimator__C': [1, 10, 100, 1000],
         'base_estimator__degree': [1, 2, 3], 'max_samples': np.arange(0.1, 0.9, 0.1),
         'max_features': np.arange(0.1, 0.9, 0.1)},
        {'base_estimator__kernel': ['rbf'], 'base_estimator__C': [1, 10, 100, 1000],
         'base_estimator__gamma': [1, 0.1, 0.01], 'max_samples': np.arange(0.1, 0.9, 0.1),
         'max_features': np.arange(0.1, 0.9, 0.1)}
    ]

    clf_knn_bag = BaggingClassifier(KNeighborsClassifier())
    parameters_knn_bag = [
        {'base_estimator__weights': ['distance'], 'base_estimator__n_neighbors': np.arange(1, 12, 1),
         'base_estimator__p': np.arange(1, 8, 1),
         'max_samples': np.arange(0.1, 0.9, 0.1), 'max_features': np.arange(0.1, 0.9, 0.1)},
        {'base_estimator__weights': ['uniform'], 'base_estimator__n_neighbors': np.arange(1, 12, 1),
         'max_samples': np.arange(0.1, 0.9, 0.1), 'max_features': np.arange(0.1, 0.9, 0.1)},
    ]

    clf_lr_bag = BaggingClassifier(LogisticRegression())
    parameters_lr_bag = [
        {
            'base_estimator__penalty': ['none'],
            'base_estimator__solver': ['sag', 'lbfgs', 'newton-cg', 'saga'],
            'base_estimator__max_iter': [10000],
            'max_samples': np.arange(0.1, 0.9, 0.1), 'max_features': np.arange(0.1, 0.9, 0.1)
        },
        {
            'base_estimator__penalty': ['l1'],
            'base_estimator__solver': ['liblinear', 'saga'],
            'base_estimator__C': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.01, 0.1, 1, 10, 100, 1000],
            'base_estimator__max_iter': [10000],
            'max_samples': np.arange(0.1, 0.9, 0.1), 'max_features': np.arange(0.1, 0.9, 0.1)
        },
        {
            'base_estimator__penalty': ['l2'],
            'base_estimator__solver': ['liblinear', 'sag', 'lbfgs', 'newton-cg', 'saga'],
            'base_estimator__C': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.01, 0.1, 1, 10, 100, 1000],
            'base_estimator__max_iter': [10000],
            'max_samples': np.arange(0.1, 0.9, 0.1), 'max_features': np.arange(0.1, 0.9, 0.1)
        },
        {
            'base_estimator__penalty': ['elasticnet'],
            'base_estimator__C': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.01, 0.1, 1, 10, 100, 1000],
            'base_estimator__max_iter': [10000],
            'max_samples': np.arange(0.1, 0.9, 0.1), 'max_features': np.arange(0.1, 0.9, 0.1)
        },
    ]

    clf_lda_bag = BaggingClassifier(LinearDiscriminantAnalysis())
    parameters_lda_bag = [
        {'base_estimator__solver': ['svd'], 'base_estimator__shrinkage': [None],
         'max_samples': np.arange(0.1, 0.9, 0.1), 'max_features': np.arange(0.1, 0.9, 0.1)},
        {'base_estimator__solver': ['lsqr'], 'base_estimator__shrinkage': ['auto', None],
         'max_samples': np.arange(0.1, 0.9, 0.1), 'max_features': np.arange(0.1, 0.9, 0.1)},
        {'base_estimator__solver': ['eigen'], 'base_estimator__shrinkage': ['auto', None],
         'max_samples': np.arange(0.1, 0.9, 0.1), 'max_features': np.arange(0.1, 0.9, 0.1)},
    ]

    clf_gb = GradientBoostingClassifier()
    parameters_gb = {
        'loss': ['deviance', 'exponential'],
        'learning_rate': np.arange(0.1, 0.9, 0.1),
        'n_estimators': np.arange(50, 100, 5),
        'subsample': ['friedman_mse', 'squared_error', 'mse', 'mae'],
        'min_samples_split': np.arange(2, 5, 1),
        'min_samples_leaf': np.arange(1, 3, 1),
        'min_weight_fraction_leaf': np.arange(0, 3, 1),
        'max_depth': np.arange(3, 10, 1),
        'max_features': ['auto', 'sqrt', 'log2']
    }

    clf_ada = AdaBoostClassifier()
    parameters_ada = [
        {
            'base_estimator': [MultinomialNB()], 'base_estimator__alpha': np.arange(1.0e-10, 1, 0.01),
            'n_estimators': np.arange(50, 100, 1), 'learning_rate': np.arange(0.1, 1, 0.05)
        },
        {
            'base_estimator': [GaussianNB()], 'base_estimator__var_smoothing': np.arange(1e-10, 1e-9, 1e-10),
            'n_estimators': np.arange(50, 100, 1), 'learning_rate': np.arange(0.1, 1, 0.05)
        },
        {
            'base_estimator': [BernoulliNB()], 'base_estimator__alpha': np.arange(1.0e-10, 1, 0.01),
            'n_estimators': np.arange(50, 100, 1), 'learning_rate': np.arange(0.1, 1, 0.05)
        },
        {
            'base_estimator': [ComplementNB()], 'base_estimator__alpha': np.arange(1.0e-10, 1, 0.01),
            'n_estimators': np.arange(50, 100, 1), 'learning_rate': np.arange(0.1, 1, 0.05)
        },
        {
            'base_estimator': [RandomForestClassifier()],
            'base_estimator__n_estimators': np.arange(75, 100, 1),
            'base_estimator__max_depth': np.arange(20, 80, 1),

            'base_estimator__min_samples_split': np.arange(2, 10, 1),  # 这两个要一起调
            'base_estimator__min_samples_leaf': np.arange(1, 10, 1),

            'base_estimator__max_features': np.arange(1, 20, 1),
            'n_estimators': np.arange(50, 100, 1), 'learning_rate': np.arange(0.1, 1, 0.05)
        },
        {
            'base_estimator': [LogisticRegression()],
            'base_estimator__penalty': ['none'],
            'base_estimator__solver': ['sag', 'lbfgs', 'newton-cg', 'saga'],
            'base_estimator__max_iter': [50, 100],
            'n_estimators': np.arange(50, 100, 1), 'learning_rate': np.arange(0.1, 1, 0.05)
        },
        {
            'base_estimator': [LogisticRegression()],
            'base_estimator__penalty': ['l2'],
            'base_estimator__solver': ['liblinear', 'sag', 'lbfgs', 'newton-cg', 'saga'],
            'base_estimator__C': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000],
            'base_estimator__max_iter': [50, 100],
            'n_estimators': np.arange(50, 100, 1), 'learning_rate': np.arange(0.1, 1, 0.05)
        },
        {
            'base_estimator': [LogisticRegression()],
            'base_estimator__penalty': ['l1'],
            'base_estimator__solver': ['liblinear', 'sag'],
            'base_estimator__C': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000],
            'base_estimator__max_iter': [50, 100],
            'n_estimators': np.arange(50, 100, 1), 'learning_rate': np.arange(0.1, 1, 0.05)
        },
        {
            'base_estimator': [LogisticRegression()],
            'base_estimator__penalty': ['elasticnet'],
            'base_estimator__C': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000],
            'base_estimator__max_iter': [50, 100],
            'n_estimators': np.arange(50, 100, 1), 'learning_rate': np.arange(0.1, 1, 0.05)
        },
        {
            'base_estimator': [DecisionTreeClassifier()],
            'base_estimator__criterion': ["gini", "entropy"],
            'base_estimator__max_depth': np.arange(1, 10, 1),
            'base_estimator__min_samples_split': np.arange(2, 20, 1),
            'base_estimator__min_samples_leaf': np.arange(1, 5, 1),
            'base_estimator__min_impurity_decrease': np.arange(0.1, 1, 0.1),
            'n_estimators': np.arange(50, 100, 1), 'learning_rate': np.arange(0.1, 1, 0.05)
        },
        {
            'base_estimator': [KNeighborsClassifier()],
            'base_estimator__weights': ['distance'],
            'base_estimator__n_neighbors': np.arange(1, 12, 1),
            'base_estimator__p': np.arange(1, 8, 1),
            'n_estimators': np.arange(50, 100, 1), 'learning_rate': np.arange(0.1, 1, 0.05)
        },
        {
            'base_estimator': [KNeighborsClassifier()],
            'base_estimator__weights': ['uniform'],
            'base_estimator__n_neighbors': np.arange(1, 12, 1),
            'n_estimators': np.arange(50, 100, 1), 'learning_rate': np.arange(0.1, 1, 0.05)
        },
        {
            'base_estimator': [LinearDiscriminantAnalysis()],
            'base_estimator__solver': ['svd'], 'base_estimator__shrinkage': [None],
            'n_estimators': np.arange(50, 100, 1), 'learning_rate': np.arange(0.1, 1, 0.05)
        },
        {
            'base_estimator': [LinearDiscriminantAnalysis()],
            'base_estimator__solver': ['lsqr'], 'base_estimator__shrinkage': ['auto', None],
            'n_estimators': np.arange(50, 100, 1), 'learning_rate': np.arange(0.1, 1, 0.05)
        },
        {
            'base_estimator': [LinearDiscriminantAnalysis()],
            'base_estimator__solver': ['eigen'], 'base_estimator__shrinkage': ['auto', None],
            'n_estimators': np.arange(50, 100, 1), 'learning_rate': np.arange(0.1, 1, 0.05)
        },
        {
            'base_estimator': [svm.SVC()],
            'base_estimator__kernel': ['linear'],
            'base_estimator__C': [1, 10, 100, 1000],
            'n_estimators': np.arange(50, 100, 1), 'learning_rate': np.arange(0.1, 1, 0.05)
        },
        {
            'base_estimator': [svm.SVC()],
            'base_estimator__kernel': ['poly'],
            'base_estimator__C': [1, 10, 100, 1000],
            'base_estimator__degree': [1, 2, 3],
            'n_estimators': np.arange(50, 100, 1), 'learning_rate': np.arange(0.1, 1, 0.05)
        },
        {
            'base_estimator': [svm.SVC()],
            'base_estimator__kernel': ['rbf'],
            'base_estimator__C': [1, 10, 100, 1000],
            'base_estimator__gamma': [1, 0.1, 0.01],
            'n_estimators': np.arange(50, 100, 1), 'learning_rate': np.arange(0.1, 1, 0.05)
        },
    ]

    estimators_list = [
        ('mnb', MultinomialNB()), ('gnb', GaussianNB()), ('bnb', BernoulliNB()), ('cnb', ComplementNB()),
        ('rf', RandomForestClassifier()), ('dt', DecisionTreeClassifier()), ('svc', svm.SVC()),
        ('lr', LogisticRegression()), ('knn', KNeighborsClassifier()), ('lda', LinearDiscriminantAnalysis())
    ]
    clf_sta = StackingClassifier(estimators=estimators_list)
    parameters_sta = {
        'final_estimator': [MultinomialNB()],
        'cv': np.arange(5, 10, 1),
        'stack_method': ['auto', 'predict_proba', 'decision_function', 'predict']
    }

    clf_vc = VotingClassifier(estimators=estimators_list, voting='soft', weights=[5, 1, 1, 4, 1, 0, 1, 1])
    parameters_vc = [
        {},
    ]

    for clf_single, parameters, name in \
            zip([clf_mnb, clf_gnb, clf_bnb, clf_cnb, clf_rf, clf_dt, clf_svm, clf_knn, clf_lr, clf_lda,
                 clf_mnb_bag, clf_gnb_bag, clf_bnb_bag, clf_cnb_bag, clf_rf_bag, clf_dt_bag, clf_svm_bag, clf_knn_bag,
                 clf_lr_bag, clf_lda_bag, clf_gb, clf_sta, clf_ada, clf_vc],
                [parameters_mnb, parameters_gnb, parameters_bnb, parameters_cnb, parameters_rf, parameters_dt,
                 parameters_svm, parameters_knn, parameters_lr, parameters_lda, parameters_mnb_bag, parameters_gnb_bag,
                 parameters_bnb_bag, parameters_cnb_bag, parameters_rf_bag, parameters_dt_bag, parameters_svm_bag,
                 parameters_knn_bag, parameters_lr_bag, parameters_lda_bag, parameters_gb, parameters_sta,
                 parameters_ada, parameters_vc],
                clf_name
                ):
        print("_" * 80)
        print("GridSearching: ")
        print(name)
        gs = GridSearchCV(clf_single, parameters, verbose=2, refit=True, cv=10, n_jobs=32)
        gs.fit(x_train, y_train)
        print(gs.best_score_)
        print(gs.best_params_)
        print(gs.score(x_train, y_train))

        print('The accuracy of classifying malware best grid in test set:', gs.score(x_test, y_test))
        pred = gs.predict(x_test)
        print("classification report:")
        print(classification_report(y_test, pred, target_names=target))


if __name__ == '__main__':
    load_features()
    pass
