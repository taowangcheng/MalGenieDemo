# 选择最佳特征
import jsonlines
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, ComplementNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn import feature_selection

from sklearn.model_selection import cross_val_score

import numpy as np

default_fields = ['name', 'topics', 'files', 'readme', 'description']
default_max_features = {'name': 500, 'topics': 500, 'files': 3000, 'readme': 1000, 'description': 2000}  # 79.9

default_input_train_files = {'name': '../data/train_test/train_raw/name.jsonl',
                             'topics': '../data/train_test/train_raw/topics.jsonl',
                             'files': '../data/train_test/train_raw/files.jsonl',
                             'readme': '../data/train_test/train_raw/readme.jsonl',
                             'description': '../data/train_test/train_raw/description.jsonl',
                             'label': '../data/train_test/train_raw/label.jsonl'}


# 数学表示
def doc2bow(train_features, max_features, key_tag):
    train_features = data_format_convert(train_features, key_tag)
    vectorizer = TfidfVectorizer(preprocessor=custom_preprocessor, tokenizer=custom_tokenizer,
                                 max_features=max_features)
    train_features = vectorizer.fit_transform(train_features).toarray()
    return train_features


# 数据格式转换
def data_format_convert(features, key_tag):
    new_features = []

    if key_tag != "name":
        for feature in features:
            temp = []
            for tokens in feature:
                for token in tokens:
                    if len(token) > 1:
                        temp.append(token)
            new_features.append(' '.join(temp))
    elif key_tag == "name":
        for feature in features:
            temp = []
            for token in feature:
                if len(token) > 1:
                    temp.append(token)
            new_features.append(' '.join(temp))
    else:
        print("key_tag值错误")
    return new_features


def custom_tokenizer(sample):
    return sample.split(' ')


def custom_preprocessor(sample):
    return sample


# 卡方选择
def chi2(x_train_field, y_train, k_feature):
    select_field = feature_selection.SelectPercentile(feature_selection.chi2, percentile=k_feature)
    x_train = select_field.fit_transform(x_train_field, y_train)
    return x_train


# 最佳特征选择
def select_best_feature(input_files=None, fields=None, fields_max_features=None, ):
    if fields is None:
        fields = default_fields
    if input_files is None:
        input_files = default_input_train_files
    if fields_max_features is None:
        fields_max_features = default_max_features

    with jsonlines.open(input_files['label']) as reader:
        y_train = list(reader)
    train_features = {}
    for train_file, max_features, key_tag in zip(list(input_files.values())[:-1],
                                                 fields_max_features.values(), fields):
        with jsonlines.open(train_file) as train_reader:
            train_features[key_tag] = doc2bow(list(train_reader), max_features, key_tag)

    x_train_name = train_features['name']
    x_train_topics = train_features['topics']
    x_train_files = train_features['files']
    x_train_readme = train_features['readme']
    x_train_description = train_features['description']

    percentiles = range(1, 100, 1)
    results_name = []
    results_topics = []
    results_files = []
    results_readme = []
    results_description = []

    # mnb_tfidf = StackingClassifier(estimators=[
    #     ('mnb', MultinomialNB()), ('gnb', GaussianNB()), ('bnb', BernoulliNB()), ('cnb', ComplementNB()),
    #     ('rf', RandomForestClassifier()), ('dt', DecisionTreeClassifier()), ('svc', svm.SVC()),
    #     ('lr', LogisticRegression()), ('knn', KNeighborsClassifier()), ('lda', LinearDiscriminantAnalysis())
    # ])
    mnb_tfidf = MultinomialNB()
    train_features_new = {}
    k1, k2, k3, k4, k5 = [50, 50, 50, 50, 50]

    len_name_1 = int(x_train_name.shape[1])
    for k_feature in percentiles:
        print(k_feature)
        k_list = [k_feature, k2, k3, k4, k5]
        for field, k in zip(fields, k_list):
            train_features_new[field] = chi2(train_features[field], y_train, k)

        x_train = np.concatenate((train_features_new['name'], train_features_new['topics'],
                                  train_features_new['files'], train_features_new['readme'],
                                  train_features_new['description']), axis=1)
        scores = cross_val_score(mnb_tfidf, x_train, y_train, cv=10, scoring='precision', n_jobs=10)
        results_name = np.append(results_name, scores.mean())
    opt = np.where(results_name == results_name.max())[0]
    length = int((opt.shape[0]) / 2)
    k1 = percentiles[opt[length - 1]]
    len_name_2 = int(len_name_1 * (int(k1)) / 100)
    print("name最佳feature的百分比为 %d，原始长度为 %d, 选择的长度为 %d" % (k1, len_name_1, len_name_2))

    len_topics_1 = int(x_train_topics.shape[1])
    for k_feature in percentiles:
        k_list = [k1, k_feature, k3, k4, k5]
        for field, k in zip(fields, k_list):
            train_features_new[field] = chi2(train_features[field], y_train, k)

        x_train = np.concatenate((train_features_new['name'], train_features_new['topics'],
                                  train_features_new['files'], train_features_new['readme'],
                                  train_features_new['description']), axis=1)
        scores = cross_val_score(mnb_tfidf, x_train, y_train, cv=10, scoring='precision', n_jobs=10)
        results_topics = np.append(results_topics, scores.mean())
    opt = np.where(results_topics == results_topics.max())[0]
    length = int((opt.shape[0]) / 2)
    k2 = percentiles[opt[length - 1]]
    len_topics_2 = int(len_topics_1 * (int(k2)) / 100)
    print("topics最佳feature的百分比为 %d，原始长度为 %d, 选择的长度为 %d" % (k2, len_topics_1, len_topics_2))

    len_files_1 = int(x_train_files.shape[1])
    for k_feature in percentiles:
        k_list = [k1, k2, k_feature, k4, k5]
        for field, k in zip(fields, k_list):
            train_features_new[field] = chi2(train_features[field], y_train, k)

        x_train = np.concatenate((train_features_new['name'], train_features_new['topics'],
                                  train_features_new['files'], train_features_new['readme'],
                                  train_features_new['description']), axis=1)
        scores = cross_val_score(mnb_tfidf, x_train, y_train, cv=10, scoring='precision', n_jobs=10)
        results_files = np.append(results_files, scores.mean())
    # print(results_files)
    opt = np.where(results_files == results_files.max())[0]
    length = int((opt.shape[0]) / 2)
    k3 = percentiles[opt[length - 1]]
    len_files_2 = int(len_files_1 * (int(k3)) / 100)
    print("files最佳feature的百分比为 %d，原始长度为 %d, 选择的长度为 %d" % (k3, len_files_1, len_files_2))

    len_readme_1 = int(x_train_readme.shape[1])
    for k_feature in percentiles:
        k_list = [k1, k2, k3, k_feature, k5]
        for field, k in zip(fields, k_list):
            train_features_new[field] = chi2(train_features[field], y_train, k)

        x_train = np.concatenate((train_features_new['name'], train_features_new['topics'],
                                  train_features_new['files'], train_features_new['readme'],
                                  train_features_new['description']), axis=1)
        scores = cross_val_score(mnb_tfidf, x_train, y_train, cv=10, scoring='precision', n_jobs=10)
        results_readme = np.append(results_readme, scores.mean())
    # print(results_readme)
    opt = np.where(results_readme == results_readme.max())[0]
    length = int((opt.shape[0]) / 2)
    k4 = percentiles[opt[length - 1]]
    len_readme_2 = int(len_readme_1 * (int(k4)) / 100)
    print("readme最佳feature的百分比为 %d，原始长度为 %d, 选择的长度为 %d" % (k4, len_readme_1, len_readme_2))

    len_description_1 = int(x_train_description.shape[1])
    for k_feature in percentiles:
        k_list = [k1, k2, k3, k4, k_feature]
        for field, k in zip(fields, k_list):
            train_features_new[field] = chi2(train_features[field], y_train, k)

        x_train = np.concatenate((train_features_new['name'], train_features_new['topics'],
                                  train_features_new['files'], train_features_new['readme'],
                                  train_features_new['description']), axis=1)
        scores = cross_val_score(mnb_tfidf, x_train, y_train, cv=10, scoring='precision', n_jobs=10)
        results_description = np.append(results_description, scores.mean())
    opt = np.where(results_description == results_description.max())[0]
    length = int((opt.shape[0]) / 2)
    k5 = percentiles[opt[length - 1]]
    len_description_2 = int(len_description_1 * (int(k5)) / 100)
    print("description最佳feature的百分比为 %d，原始长度为 %d, 选择的长度为 %d" % (k5, len_description_1, len_description_2))


if __name__ == '__main__':
    select_best_feature()

