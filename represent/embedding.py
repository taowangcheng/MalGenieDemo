import gensim.models
import tempfile
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import math
import random
from sklearn.decomposition import PCA


class MyCorpus:
    def __init__(self, train_features, key_tag):
        self.train_features = train_features
        self.key_tag = key_tag

    def __iter__(self):
        for feature in self.train_features:
            for tokens in feature:
                yield tokens


def word2vec(train_features, test_features, real_features, vector_size, min_count, class_mode, represent_mode, key_tag):
    # 数据格式转换
    train_features = data_format_convert(train_features, key_tag)
    test_features = data_format_convert(test_features, key_tag)
    real_features = data_format_convert(real_features, key_tag)
    
    # 训练模型
    model = gensim.models.Word2Vec(sentences=MyCorpus(train_features, key_tag), vector_size=vector_size,
                                   min_count=min_count)
    # 保存模型
    save_model(model, class_mode, represent_mode, key_tag)
    fw = open(f"../data/models/word2vec/{class_mode}-{represent_mode}-{key_tag}_error.log", "w")
    train_features = sentence2vec(model, train_features, vector_size, fw, represent_mode, flag="train set")
    test_features = sentence2vec(model, test_features, vector_size, fw, represent_mode, flag="train_test set")
    real_features = sentence2vec(model, real_features, vector_size, fw, represent_mode, flag="train_real set")

    return train_features, test_features, real_features


# 数据格式转换
def data_format_convert(features, key_tag):
    if key_tag == "name":
        new_features = []
        for feature in features:
            temp = [feature]
            new_features.append(temp)
        features = new_features

    new_features = []
    for feature in features:
        temp_feature = []
        for tokens in feature:
            temp_tokens = []
            for token in tokens:
                if len(token) > 1:
                    temp_tokens.append(token)
            temp_feature.append(temp_tokens)
        new_features.append(temp_feature)
    return new_features


def get_weight(frequency):
    return 1 / math.log(frequency)


def sentence2vec(model, features, vector_size, fw, mode, flag):
    vectors = np.zeros((len(features), vector_size), dtype=np.float32)
    fw.write(f"=====================\n{flag}\n====================\n")
    for index, feature in enumerate(features):
        temp_vectors = [np.zeros(vector_size)]
        temp_weights = [0]
        for tokens in feature:
            for token in tokens:
                try:
                    temp_vectors.append(model.wv[token])
                    temp_weights.append(get_weight(model.wv.get_vecattr(token, "count")))
                except KeyError:
                    fw.write("The word " + token + " does not appear in this model\n")
        if mode == 'mode3':
            # 向量平均
            vectors[index] = np.asarray(temp_vectors).mean(axis=0)
        elif mode == 'mode4' or mode == 'mode5':
            if np.sum(temp_weights) != 0:
                vectors[index] = np.average(np.asarray(temp_vectors), axis=0, weights=temp_weights)
            else:
                print(f"{feature} word2vec为全零向量")
        else:
            print("mode值错误")
    if mode == 'mode5':
        sentence_set = list(vectors.copy())
        # calculate PCA of this sentence set
        if len(features) < vector_size:
            pca_size = len(features)
        else:
            pca_size = vector_size
        pca = PCA(n_components=pca_size)
        pca.fit(np.array(sentence_set))
        u = pca.components_[0]  # the PCA vector
        u = np.multiply(u, np.transpose(u))  # u x uT

        # pad the vector?  (occurs if we have less sentences than embeddings_size)
        if len(u) < vector_size:
            for i in range(vector_size - len(u)):
                u = np.append(u, 0)  # add needed extension for multiplication below

        # resulting sentence vectors, vs = vs -u x uT x vs
        for index, vs in enumerate(sentence_set):
            sub = np.multiply(u, vs)
            vectors[index] = np.subtract(vs, sub)

    return vectors


# 模型加载
def load_model(mode, key_tag):
    fr = open(f"../data/models/word2vec/{mode}-{key_tag}.txt", "r")
    file = fr.readline().strip()
    model = gensim.models.Word2Vec.load(file)
    return model


# 模型保存
def save_model(model, class_mode, represent_mode, key_tag):
    file_name = 'gensim-model-' + key_tag
    fw = open(f"../data/models/word2vec/{class_mode}-{represent_mode}-{key_tag}.txt", "w")
    with tempfile.NamedTemporaryFile(prefix=file_name, delete=False, dir='../data/models/word2vec') as tmp:
        temporary_filepath = tmp.name
        fw.write(temporary_filepath)
        model.save(temporary_filepath)
    fw.close()


def display(model):
    x_vals, y_vals, labels = reduce_dimensions(model)
    plot_function = plot_with_matplotlib
    print(model.wv.most_similar(positive=['hacking'], topn=20))
    plot_function(x_vals, y_vals, labels)


def reduce_dimensions(model):
    num_dimensions = 2  # final num dimensions (2D, 3D, etc)
    # extract the words & their vectors, as numpy arrays
    vectors = np.asarray(model.wv.vectors)
    labels = np.asarray(model.wv.index_to_key)  # fixed-width numpy strings
    # reduce using t-SNE
    tsne = TSNE(n_components=num_dimensions, random_state=0)
    vectors = tsne.fit_transform(vectors)
    x_vals = [v[0] for v in vectors]
    y_vals = [v[1] for v in vectors]
    return x_vals, y_vals, labels


def plot_with_matplotlib(x_vals, y_vals, labels):
    random.seed(0)

    plt.figure(figsize=(12, 12))
    plt.scatter(x_vals, y_vals, alpha=0.2)

    #
    # Label randomly subsampled 25 data points
    #
    indices = list(range(len(labels)))
    selected_indices = random.sample(indices, 250)
    for i in selected_indices:
        plt.annotate(labels[i], (x_vals[i], y_vals[i]))
    plt.show()
