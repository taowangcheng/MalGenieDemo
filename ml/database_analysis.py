import numpy as np
from cleanlab.filter import find_label_issues
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.naive_bayes import MultinomialNB
from cleanlab.pruning import get_noise_indices
from sklearn.model_selection import train_test_split
import jsonlines


if __name__ == '__main__':

    database = np.load('E:/sklearn_tfidf_vector.npz')

    X_train = database['x_train']
    X_test = database['x_test']
    y_train = database['train_label']
    y_test = database['test_label']
    target_names = ['good', 'malware']

    clf_mnb = MultinomialNB()

    # clf_mnb.fit(X_train, y_train)

    # pred = clf_mnb.predict_proba(X_train)  # 分别属于每个类别的概率
    kf = KFold(n_splits=10)
    for train, test in kf.split(X_train):
        clf_mnb.fit(np.array(X_train)[train], np.array(y_train)[train])
    pred = clf_mnb.predict_proba(X_train)  # 分别属于每个类别的概率
    # y_pred = cross_val_predict.predictions(clf_mnb, X_train, y_train, cv=10)  # 数据集中标记的归类
    y_pred = y_train  # 数据集中标记的归类

    for i in range(len(y_pred)):
        print(i)
        print(pred[i])
        print(y_pred[i])
        print()
    ordered_label_issues = find_label_issues(
        labels=y_pred,
        pred_probs=pred,
        return_indices_ranked_by='self_confidence',
    )
    print(ordered_label_issues)

    bowed_repo = []
    X = []
    y = []
    route_bowed = ['C:\\Users\\chenxuyang\\Desktop\\word2vec_after.jsonl']    # 读取分词后的文件

    route_bowed_name = []
    route_bowed_files = []
    route_bowed_topics = []
    route_bowed_description = []
    route_bowed_readme = []

    X_train_topics = []
    X_train_description = []
    X_train_readme = []
    X_train_files = []
    X_train_name = []


    X_test_topics = []
    X_test_description = []
    X_test_readme = []
    X_test_files = []
    X_test_name = []

    for filename in route_bowed:
        with jsonlines.open(filename) as reader:
            for repo in reader:
                bowed_repo.append(repo)
                y.append(int(repo["binary_label"]) - 1)

    for repo_dict in bowed_repo:
        repo_dict["topics"] = [token for tokens in repo_dict["topics"] for temp in tokens for token in temp]
        repo_dict["description"] = [token for tokens in repo_dict["description"] for token in tokens]
        repo_dict["readme"] = [token for tokens in repo_dict["readme"] for token in tokens]
        repo_dict["files"] = [token for tokens in repo_dict["files"] for temp in tokens for token in temp]
        repo_dict["name"] = [token for tokens in repo_dict["name"] for token in tokens]


    for i in range(len(bowed_repo)):
        X.append(str(i))

    X_train_pre, X_test_pre, y_train, y_test = train_test_split(X, y, test_size=0.26, random_state=33)
    for i in ordered_label_issues:
        print(bowed_repo[int(X_train_pre[i])])

# ######################################################################################################################

# from sklearn.linear_model import LogisticRegression
# from cleanlab.classification import CleanLearning
#
# clf = LogisticRegression() # any classifier implementing the sklearn API
# cl = CleanLearning(clf=clf)
# cl.fit(X=X, labels=labels)