import pickle
import numpy as np
import jsonlines
import sys
sys.path.append('./represent/')


default_fields = ['name', 'topics', 'files', 'readme', 'description']

default_input_real_files = {'name': './data/real/preprocess/name.jsonl',
                            'topics': './data/real/preprocess/topics.jsonl',
                            'files': './data/real/preprocess/files.jsonl',
                            'readme': './data/real/preprocess/readme.jsonl',
                            'description': './data/real/preprocess/description.jsonl'
                            }
model_tfidf_files = {'name': './data/model/tfidf/name.pkl',
                     'topics': './data/model/tfidf/topics.pkl',
                     'files': './data/model/tfidf/files.pkl',
                     'readme': './data/model/tfidf/readme.pkl',
                     'description': './data/model/tfidf/description.pkl',
                     }
model_chi2_files = {'name': './data/model/chi2/name.pkl',
                    'topics': './data/model/chi2/topics.pkl',
                    'files': './data/model/chi2/files.pkl',
                    'readme': './data/model/chi2/readme.pkl',
                    'description': './data/model/chi2/description.pkl',
                    }
real_features = {}


def doc2bow(features, key_tag):
    features = data_format_convert(features, key_tag)

    vectorizer = pickle.load(open(model_tfidf_files[key_tag], "rb"))
    features_pre = vectorizer.transform(features).toarray()

    vectorizer_chi2 = pickle.load(open(model_chi2_files[key_tag], "rb"))
    real = vectorizer_chi2.transform(features_pre)

    return real


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


def represent(fields=None):
    if fields is None:
        fields = default_fields
    for key_tag in fields:
        with jsonlines.open(default_input_real_files[key_tag]) as reader:
            real = list(reader)
            real_features[f"{key_tag}"] = doc2bow(real, key_tag)
    real = np.concatenate((real_features[fields[0]], real_features[fields[1]], real_features[fields[2]],
                           real_features[fields[3]], real_features[fields[4]]), axis=1)
    return real


def ml():
    x_real = represent()
    clf = pickle.load(open("./data/model/ml/stacking.pkl", 'rb'))
    real_pred = clf.predict(x_real)
    index_list = []
    for index, x in enumerate(real_pred):
        if x == 1:
            index_list.append(index)
    print(index_list)

ml()

