import jsonlines
import numpy as np
import json
import bow
import embedding
import fast_text

# from . import bow
# from . import embedding
# from . import fast_text

import pandas as pd

# best k : 30, 10, 100, 10, 400
# 5个字段
default_fields = ['name', 'topics', 'files', 'readme', 'description']
# default_max_features = {'name': 100, 'topics': 50, 'files': 200, 'readme': 50, 'description': 1000}  # 79.9
# default_max_features = {'name': 500, 'topics': 100, 'files': 2000, 'readme': 1000, 'description': 1000}  # 79.9
# related to malicious  tfidf 0.8196428571428571

# 不同字段的最大词数
default_max_features = {'name': 500, 'topics': 500, 'files': 3000, 'readme': 1000, 'description': 2000}  # 79.9

# default_percentiles = {'name': 30, 'topics': 20, 'files': 50, 'readme': 20, 'description': 40}
# default_percentiles = {'name': 30, 'topics': 10, 'files': 100, 'readme': 10, 'description': 400}  # kbest

# K方选择后各字段的占比
default_percentiles = {'name': 66, 'topics': 97, 'files': 31, 'readme': 9, 'description': 59}

# 向量长度
default_vector_sizes = {'name': 128, 'topics': 128, 'files': 128, 'readme': 128, 'description': 128}
default_min_counts = {'name': 3, 'topics': 2, 'files': 5, 'readme': 3, 'description': 10}

# 分类方式
default_classify_modes = {
    'mode1': "two categories",
}

# 数据表示方式
default_represent_modes = {
    'mode2': "tfidf",
}

# 文件列表
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

default_output_file = "../data/train_test/train_test.npz"

output_save = {'name': '../data/model/tfidf/name.pkl',
               'topics': '../data/model/tfidf/topics.pkl',
               'files': '../data/model/tfidf/files.pkl',
               'readme': '../data/model/tfidf/readme.pkl',
               'description': '../data/model/tfidf/description.pkl',
               'label': '../data/train_test/real_raw/label.jsonl'
               }
output_chi2_file = {'name': '../data/model/chi2/name.pkl',
               'topics': '../data/model/chi2/topics.pkl',
               'files': '../data/model/chi2/files.pkl',
               'readme': '../data/model/chi2/readme.pkl',
               'description': '../data/model/chi2/description.pkl',
               'label': '../data/train_test/real_raw/label.jsonl'
               }

def process(fields=None, input_files=None, output_file=None, classify_modes=None,
            represent_modes=None, fields_max_features=None,
            vector_sizes=None, min_counts=None, fields_percentiles=None):
    if fields is None:
        fields = default_fields
    if input_files is None:
        input_files = [default_input_train_files, default_input_test_files, default_input_real_files]
    if output_file is None:
        output_file = default_output_file
    if classify_modes is None:
        classify_modes = default_classify_modes
    if represent_modes is None:
        represent_modes = default_represent_modes
    if fields_max_features is None:
        fields_max_features = default_max_features
    if vector_sizes is None:
        vector_sizes = default_vector_sizes
    if min_counts is None:
        min_counts = default_min_counts
    if fields_percentiles is None:
        fields_percentiles = default_percentiles

    # def tree():
    #     return defaultdict(tree)
    # tree = lambda: defaultdict(tree)
    data_dict = {}

    # 模式选择
    for class_mode in classify_modes.keys():
        for represent_mode in represent_modes.keys():

            # 打开标签文件
            for train_target_file, test_target_file, real_target_file in zip(list(input_files[0].values())[-1:],
                                                                             list(input_files[1].values())[-1:],
                                                                             list(input_files[2].values())[-1:]):
                with jsonlines.open(train_target_file) as train_target_reader:
                    with jsonlines.open(test_target_file) as test_target_reader:
                        with jsonlines.open(real_target_file) as real_target_reader:
                            train_target = list(train_target_reader)
                            test_target = list(test_target_reader)
                            real_target = list(real_target_reader)

                            train_target, train_indexes = choose_mode(train_target, class_mode)
                            test_target, test_indexes = choose_mode(test_target, class_mode)
                            real_target, real_indexes = choose_mode(real_target, class_mode)

                            # 打开特征文件
                            for train_file, test_file, real_file, output,output_chi2, max_features, percentile, vector_size, min_count, key_tag \
                                    in zip(list(input_files[0].values())[:-1], list(input_files[1].values())[:-1],
                                           list(input_files[2].values())[:-1], list(output_save.values())[:-1],list(output_chi2_file.values())[:-1],
                                           fields_max_features.values(),
                                           fields_percentiles.values(), vector_sizes.values(), min_counts.values(),
                                           fields):
                                with jsonlines.open(train_file) as train_reader:
                                    with jsonlines.open(test_file) as test_reader:
                                        with jsonlines.open(real_file) as real_reader:
                                            train_features = list(
                                                np.asarray(list(train_reader), dtype=object)[train_indexes])
                                            test_features = list(
                                                np.asarray(list(test_reader), dtype=object)[test_indexes])
                                            real_features = list(
                                                np.asarray(list(real_reader), dtype=object)[real_indexes])

                                            train_features, test_features, real_features = feature_represent(
                                                train_features, train_target,
                                                test_features, real_features, max_features,
                                                percentile, represent_mode, key_tag, output,output_chi2)
                                            # 存储
                                            data_dict[f"{class_mode}-{represent_mode}-train-label"] = np.asarray(
                                                train_target)
                                            data_dict[f"{class_mode}-{represent_mode}-test-label"] = np.asarray(
                                                test_target)
                                            data_dict[f"{class_mode}-{represent_mode}-real-label"] = np.asarray(
                                                real_target)

                                            data_dict[f"{class_mode}-{represent_mode}-train-{key_tag}"] = train_features
                                            data_dict[f"{class_mode}-{represent_mode}-test-{key_tag}"] = test_features
                                            data_dict[f"{class_mode}-{represent_mode}-real-{key_tag}"] = real_features
    save_features(output_file, data_dict)


def save_features(filename, data_dict):
    np.savez(filename, data_dict=data_dict)


# 数学表示
def feature_represent(train_features, train_target, test_features, real_features, max_features, percentile,
                      represent_mode, key_tag, output,output_chi2):
    if represent_mode == 'mode1' or represent_mode == 'mode2':
        train_features, test_features, real_features = bow.doc2bow(train_features, train_target,
                                                                   test_features, real_features, max_features,
                                                                   percentile, represent_mode, key_tag, output,output_chi2)
        print(output, "已保存")
    return train_features, test_features, real_features


# 分类方式选择
def choose_mode(labels, mode):
    new_labels = []
    indexes = []
    for index, label in enumerate(labels):
        if mode == 'mode1':
            if label == 0 or label == 1:
                new_labels.append(label)
                indexes.append(index)
    return new_labels, indexes


if __name__ == '__main__':
    process()
    # save_vector(default_classify_modes, default_fields)
