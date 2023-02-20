import streamlit as st
import time
import pickle
import pandas as pd
import numpy as np
import jsonlines
import sys
sys.path.append('represent/')
import os
os.system("apt install python-enchant")
import random
import re
import enchant
from spiral import ronin
import contractions
from enchant.tokenize import get_tokenizer, HTMLChunker, EmailFilter, URLFilter
from nltk import WordNetLemmatizer, pos_tag
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk import sent_tokenize

st.set_page_config(page_title="MalMine", page_icon="📈", layout="wide")


default_fields = ['name', 'topics', 'files', 'readme', 'description']

modelname_tfidf_files = {'name': 'data/model/tfidf/name.pkl',
                     'topics': 'data/model/tfidf/topics.pkl',
                     'files': 'data/model/tfidf/files.pkl',
                     'readme': 'data/model/tfidf/readme.pkl',
                     'description': 'data/model/tfidf/description.pkl',
                     }
modelname_chi2_files = {'name': 'data/model/chi2/name.pkl',
                    'topics': 'data/model/chi2/topics.pkl',
                    'files': 'data/model/chi2/files.pkl',
                    'readme': 'data/model/chi2/readme.pkl',
                    'description': 'data/model/chi2/description.pkl',
                    }


# 模型加载
@st.experimental_singleton
def load_models():
    model_tfidf = {}
    model_chi2 = {}
    for field in default_fields:
        model_tfidf[field] = pickle.load(open(modelname_tfidf_files[field], "rb"))
        model_chi2[field] = pickle.load(open(modelname_chi2_files[field], "rb"))
    clf = pickle.load(open("data/model/ml/stacking.pkl", 'rb'))
    return model_tfidf, model_chi2, clf

model_tfidf, model_chi2, clf = load_models()



# 数据预处理
def exec_process(input_files=None, output_files=None):
    if input_files is None:
        input_files = ["data/real/raw/repos_feature.jsonl"]
    if output_files is None:
        output_files = ["data/real/preprocess/repos_real.jsonl"]
    for input_file, output_file in zip(input_files, output_files):
        with jsonlines.open(input_file) as reader:
            with jsonlines.open(output_file, 'w') as writer:
                repo_dicts = list(reader)
                random.shuffle(repo_dicts)
                random.shuffle(repo_dicts)
                random.shuffle(repo_dicts)

                # 不同字段采取不同处理方式
                for repo_dict in repo_dicts:
                    repo_dict["name"] = name_process(repo_dict["name"])
                    repo_dict["topics"] = topics_process(repo_dict["topics"])

                    repo_dict["files"] = files_process(repo_dict["files"])

                    repo_dict["description"] = description_process(repo_dict["description"])
                    repo_dict["readme"] = readme_process(repo_dict["readme"])
                    repo_dict["label"] = 0
                    writer.write(repo_dict)

# 各字段处�?
def name_process(name: str):
    return short_text_process(name)


def topics_process(topics: list):
    return_list = []
    for topic in topics:
        return_list.append(short_text_process(topic))
    return return_list


def files_process(files: list):
    return_list = []
    for file in files:
        return_list.append(short_text_process(file))
    return return_list


def description_process(descr: str):
    return long_text_process(descr)


def readme_process(readme: str):
    return long_text_process(readme)



# 短文本切分
delim_chars_replace = re.compile(r"[,._\-<>?;\':\"{}\[\]|\\=+*~`!@#$%^&()]+")
# nums_filter = re.compile(r'[0-9]+')
blank_char_filter = re.compile(r'\s{2,}')


# dict_chk = enchant.DictWithPWL("en_US", "data/wiki_words/big_word_dictionary.txt")

dict_chk = enchant.DictWithPWL("en_US", r"D:\JetBrains\Python\WorkSpace\demo\data\wiki_words\big_word_dictionary.txt")

def short_transform(word: str):
    # 该函数是完成数字替换字母的函数
    transform_dict = {'0': 'o', '1': 'i', '2': 'z', '3': 'e', '4': 'a', '5': 's', '6': '', '7': '', '8': 'b', '9': 'p'}
    new_word = ""
    for index, letter in enumerate(word):
        if letter.isdigit():
            if index == 0 or index == len(word) - 1:
                pass  # 如果后面还是数字就过滤掉这个东西
            else:
                if word[index + 1].isalpha() and word[index - 1].isalpha():
                    new_word += transform_dict[str(letter)]
                else:
                    pass
        else:
            new_word += letter
    return new_word


def check_single_char(word: str):
    flag = 1
    flag1 = 0
    for index, letter in enumerate(word):
        if flag1 == 0:
            if letter.isalpha():
                flag1 = 1
            else:
                flag = 0
                break
        else:
            if letter == '-':
                flag1 = 0
            else:
                flag = 0
                break
    if flag == 0:
        return False
    else:
        return True


def short_text_process(name: str):
    name = name.encode('ascii', 'ignore').decode()
    if check_single_char(name):
        return [delim_chars_replace.sub('', name).lower()]

    name = delim_chars_replace.sub(' ', name)  # 将特殊字符转化成空格
    name = short_transform(name)  # 进行数字的剔�?
    name = name.strip()
    name = blank_char_filter.sub(' ', name)
    word_list = []  # 最后切分的结果
    if name == '' or name == ' ':
        return []
    if name != ' ':
        temp_word_list = name.split(' ')
        for word in temp_word_list:
            if word.istitle():
                word_list.append(word.lower())
                continue
            if word.islower() or word.isupper():  # 先判断字符是否是纯大写或者纯小写，或者是只有首字母大�?
                if dict_chk.check(word.lower()):
                    word_list.append(word.lower())  # 如果直接是单词，直接添加，否则就直接切分
                else:
                    word_list += ronin.split(word.lower())
            else:
                index_last = 0
                index_flag = 0
                if len(word) > 1:
                    word = word[0].lower() + word[1:]
                else:
                    word = word.lower()
                temp_temp_word_list = []  # 新建一个暂时存储word切分结果的list
                for index, letter in enumerate(word):
                    if letter.isupper():  # 说明当前没有进入连续大写字母的情况
                        if index != index_last + 1 and not index_flag:
                            temp_temp_word_list.append(word[index_last:index].lower())
                            index_last = index
                        else:
                            index_flag = 1
                    else:
                        if index_flag:
                            index_flag = False
                            temp_temp_word_list.append(word[index_last:index - 1].lower())
                            index_last = index - 1
                temp_temp_word_list.append(word[index_last:].lower())  # 到现在为止，利用我们自己的方法将这个词汇进行了驼峰切分
                num = 0
                max_num = 2
                for temp_word in temp_temp_word_list:
                    if not dict_chk.check(temp_word.lower()):
                        num += 1
                if num <= max_num:
                    word_list += temp_temp_word_list
                else:
                    word_list += ronin.split(word.lower())
    return word_list



# 长文本切分
code_block_replace = re.compile(r'(`+)(.*)(`+)', re.S)
image_block_replace = re.compile(r'!\[(.*)]\((.*)\)', re.S)

# 专有名词处理
# from nltk import MWETokenizer
# proper_nouns = [('spring', 'boot')]
# multi_word_tk = MWETokenizer(proper_nouns, separator=' ')

# 分词器附加html提取，email过滤，url过滤
word_tk = get_tokenizer("en_US", chunkers=(HTMLChunker,), filters=(EmailFilter, URLFilter))
word_lemma = WordNetLemmatizer()
stop_words = stopwords.words('english')
stop_words.append('cannot')
# stop_words.append('youtube')
stop_words = set(stop_words)  # 提高查询速度


# 词性获取
def get_word_tag(word_tag):
    if word_tag.startswith('J'):
        return wordnet.ADJ
    elif word_tag.startswith('V'):
        return wordnet.VERB
    elif word_tag.startswith('N'):
        return wordnet.NOUN
    elif word_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


def long_text_process(doc):
    if doc:
        doc = code_block_replace.sub('\n', doc)
        doc = image_block_replace.sub('\t', doc)
        # doc = doc.encode('ascii', 'ignore').decode()
        doc = doc.encode('ascii', 'ignore').decode().replace("http", " http")
        doc = contractions.fix(doc, slang=True)     # 将缩略词展开
        doc = sent_tokenize(doc.lower())
        word_list = []
        for sentence in doc:
            sentence_temp = []
            for (word, pos) in word_tk(sentence):
                sentence_temp.append(word)
            sentence = sentence_temp
            # sentence = multi_word_tk.tokenize(sentence)  # 拆分之后将专有词拼接
            tagged_sent = pos_tag(sentence)  # 获取单词词性
            lemmaed_sent = []
            for tag in tagged_sent:
                wordnet_pos = get_word_tag(tag[1]) or wordnet.NOUN  # 按位运算
                lemmaed_sent.append(word_lemma.lemmatize(tag[0], pos=wordnet_pos))  # 词形还原
            sentence = [word for word in lemmaed_sent if word not in stop_words]
            sentence_temp = []
            for word in sentence:
                temp = word.split("'")
                sentence_temp += temp
            sentence_final = [word for word in sentence_temp if len(word) > 1]
            word_list.append(sentence_final)
    else:
        word_list = [[]]
    return word_list


# 对分词后的文件进行分割字段的处理
# 加载文件的位置
default_input_real_split = "data/real/preprocess/repos_real.jsonl"

default_real_files_split = {'name': 'data/real/preprocess/name.jsonl',
                      'topics': 'data/real/preprocess/topics.jsonl',
                      'files': 'data/real/preprocess/files.jsonl',
                      'readme': 'data/real/preprocess/readme.jsonl',
                      'description': 'data/real/preprocess/description.jsonl',
}


# 对于真实数据集的处理
def real_repo_split(input_file=default_input_real_split, real_files=None):
    fields = ['name', 'topics', 'files', 'readme', 'description']
    if real_files is None:
        real_files = default_real_files_split
    with jsonlines.open(input_file) as reader:
        real_repos = list(reader)
        real_split = pd.DataFrame(real_repos)
        # real["label"] = pd.DataFrame([label for label in real["binary_label"]])
        for field in fields:
            with jsonlines.open(real_files[field], 'w') as real_writer:
                real_writer.write_all(real_split[field])



default_input_real_files = {'name': 'data/real/preprocess/name.jsonl',
                            'topics': 'data/real/preprocess/topics.jsonl',
                            'files': 'data/real/preprocess/files.jsonl',
                            'readme': 'data/real/preprocess/readme.jsonl',
                            'description': 'data/real/preprocess/description.jsonl'
                            }
real_features = {}


# Tf-idf
def doc2bow(features, key_tag, model_tfidf, model_chi2):
    features = data_format_convert(features, key_tag)

    vectorizer = model_tfidf[key_tag]
    features_pre = vectorizer.transform(features).toarray()

    vectorizer_chi2 = model_chi2[key_tag]
    real = vectorizer_chi2.transform(features_pre)

    return real


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


# 文本表示
def represent(fields=None):
    if fields is None:
        fields = default_fields
    for key_tag in fields:
        with jsonlines.open(default_input_real_files[key_tag]) as reader:
            real = list(reader)
            real_features[f"{key_tag}"] = doc2bow(real, key_tag, model_tfidf, model_chi2)
    real = np.concatenate((real_features[fields[0]], real_features[fields[1]], real_features[fields[2]],
                           real_features[fields[3]], real_features[fields[4]]), axis=1)
    return real


def ml(clf, x_real):
    mal_repos = []
    good_repos = []

    real_pred = clf.predict(x_real)
    index_list_mal = []
    index_list_good = []
    for index, x in enumerate(real_pred):
        if x == 1:
            index_list_mal.append(index)
        if x == 0:
            index_list_good.append(index)
    with jsonlines.open("data/real/preprocess/repos_real.jsonl") as reader:
        real_repos = list(reader)
        for index in index_list_mal:
            mal_repos.append(real_repos[index]['full_name'])
        for index in index_list_good:
            good_repos.append(real_repos[index]['full_name'])
        return mal_repos, good_repos



def exec_ml():
    repos = st.session_state.repos_input.split('\n')
    with open("repos_list.txt", "w") as f:
                for repo in repos:
                    f.write(repo)
                    f.write('\n')
    with temp:
        with st.spinner('正在爬虫...'):
            os.chdir("D:\JetBrains\Python\WorkSpace\demo\spider\FeatureSpider")
            os.system("scrapy crawl feature")
            print("-----------------------------------------")
            print("爬虫结束")
    with temp:
        with st.spinner('预处理中...'):
            os.chdir("D:\JetBrains\Python\WorkSpace\demo\\")
            exec_process()
            real_repo_split()
            print("-----------------------------------------")
            time.sleep(2)
    with temp:
        with st.spinner('进行文本表示...'):
            x_real = represent()
            time.sleep(2)
    with temp:
        with st.spinner('正在分类...'):        
            mal_repos, good_repos = ml(clf, x_real)
            malware_repos = '\n'.join(mal_repos)
            benign_repos = '\n'.join(good_repos)
            time.sleep(2)

    st.session_state.repos_benign_output = benign_repos
    st.session_state.repos_malware_output = malware_repos


cols = st.columns(6)
cols[0].caption("By Ilove510")
cols[-1].caption("第十五届信息安全作品赛")

st.write("# MalMine")
st.write("""**MalMine识别Github平台的恶意软件源代码仓库**""")

st.subheader("贡献", anchor="贡献")
line1, line2, line3 = st.columns(3)
line1.markdown("""
###### 分析代码开源平台的开源工具
绕过Github API的限制  
多类型、多层次的自然语言处理方案
""")
line2.markdown("""
###### 优秀的基线模型
进行大量横向对比实验  
考虑多种集成模型与非集成模型
""")
line3.markdown("""
###### 大型开源数据集
""")
line3.metric(label="恶意软件源代码仓库数量", value="33609 个", delta="347.8 %")


st.subheader("在线演示", anchor="在线演示")
st.write("""我们在这里提供了一个MalMine主体部分的在线演示，只需要输入几个仓库，MalMine会识别出恶意仓库与良性仓库""")
repos_input = st.text_area('输入示例', 
'''Stealerium/Stealerium
Ovusi/The_Subway
GigaZero/HiveMind-JS-BotNet
ssmore98/Logistic-Regression
NicknamedCoco/GetVirusData
onezisone/sms_neural''',
            key="example", height=175, placeholder="示例输入")
repos_input = st.text_area('请输入待分类仓库',
            key="repos_input", height=175,placeholder="请输入仓库")


exec_button = st.button("执行", key="exec_button", on_click=exec_ml)
temp = st.empty()
repos_benign_output = st.text_area('输出的良性仓库', key="repos_benign_output", height=175)
repos_malware_output = st.text_area('输出的恶意仓库', key="repos_malware_output", height=175)
