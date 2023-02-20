import contractions
import re
from enchant.tokenize import get_tokenizer, HTMLChunker, EmailFilter, URLFilter
from nltk import WordNetLemmatizer, pos_tag
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk import sent_tokenize

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


def process(doc):
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


if __name__ == '__main__':
    texts = ["![这是图片](/assets/img/philly-magic-garden.jpg \"Magic Gardens\")",
             " 🚀 The ultimate project builder for Astro  🚀                                                                                      ⚠️ WARNING: In development ⚠️"]
    for text in texts:
        print(text, ": ", process(text))