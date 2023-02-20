from sklearn.model_selection import train_test_split
import jsonlines
import pandas as pd

# 加载文件的位置
default_input_real = "../data/real/preprocess/repos_real.jsonl"

default_real_files = {'name': '../data/real/preprocess/name.jsonl',
                      'topics': '../data/real/preprocess/topics.jsonl',
                      'files': '../data/real/preprocess/files.jsonl',
                      'readme': '../data/real/preprocess/readme.jsonl',
                      'description': '../data/real/preprocess/description.jsonl',
}

fields = ['name', 'topics', 'files', 'readme', 'description']


# 对于真实数据集的处理
def real_repo(input_file=default_input_real, real_files=None):
    if real_files is None:
        real_files = default_real_files
    with jsonlines.open(input_file) as reader:
        real_repos = list(reader)
        real = pd.DataFrame(real_repos)
        # real["label"] = pd.DataFrame([label for label in real["binary_label"]])
        for field in fields:
            print(field)
            with jsonlines.open(real_files[field], 'w') as real_writer:
                real_writer.write_all(real[field])


if __name__ == '__main__':
    real_repo()
