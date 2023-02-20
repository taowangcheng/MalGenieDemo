import re
import enchant
from spiral import ronin


delim_chars_replace = re.compile(r"[,._\-<>?;\':\"{}\[\]|\\=+*~`!@#$%^&()]+")
# nums_filter = re.compile(r'[0-9]+')
blank_char_filter = re.compile(r'\s{2,}')


dict_chk = enchant.DictWithPWL("en_US", "../data/wiki_words/big_word_dictionary.txt")


def transform(word: str):
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


def process(name: str):
    name = name.encode('ascii', 'ignore').decode()
    if check_single_char(name):
        return [delim_chars_replace.sub('', name).lower()]

    name = delim_chars_replace.sub(' ', name)  # 将特殊字符转化成空格
    name = transform(name)  # 进行数字的剔除
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
            if word.islower() or word.isupper():  # 先判断字符是否是纯大写或者纯小写，或者是只有首字母大写
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


if __name__ == '__main__':
    texts = ['', 'H-e-l-l-o', "thirtyDaysOfCode"]
    for text in texts:
        print(text, ": ", process(text))
        print(text, ": ", dict_chk.check(text))
