from sklearn import feature_selection
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

def doc2bow(train_features, train_target, test_features, real_features, max_features, percentile, mode, key_tag, output,output_chi2):
    train_features = data_format_convert(train_features, key_tag)
    test_features = data_format_convert(test_features, key_tag)
    real_features = data_format_convert(real_features, key_tag)

    if mode == 'mode1':
        vectorizer = CountVectorizer(preprocessor=custom_preprocessor, tokenizer=custom_tokenizer,
                                     max_features=max_features)
    elif mode == 'mode2':
        vectorizer = TfidfVectorizer(preprocessor=custom_preprocessor, tokenizer=custom_tokenizer,
                                     max_features=max_features, decode_error='replace')
    else:
        print("mode值错误")
        return
    train_features = vectorizer.fit_transform(train_features).toarray()
    pickle.dump(vectorizer, open(output, "wb"))
    test_features = vectorizer.transform(test_features).toarray()
    real_features = vectorizer.transform(real_features).toarray()

    # 卡方选择
    train_features, test_features, real_features = chi2_select(train_features, train_target, test_features,
                                                               real_features, percentile,output_chi2)

    return train_features, test_features, real_features


def chi2_select(train_features, train_target, test_features, real_features, percentile,output_chi2):
    select = feature_selection.SelectPercentile(feature_selection.chi2, percentile=percentile)
    # select = feature_selection.SelectKBest(feature_selection.chi2, k=percentile)
    new_train_features = select.fit_transform(train_features, train_target)
    pickle.dump(select, open(output_chi2, "wb"))
    new_test_features = select.transform(test_features)
    new_real_features = select.transform(real_features)

    return new_train_features, new_test_features, new_real_features


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


if __name__ == "__main__":
    pass
