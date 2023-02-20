import random

import jsonlines
import preprocess
import split_label_repos

# 文件位置
default_input_real = "../data/real/raw/repos_feature.jsonl"
default_output_real = "../data/real/preprocess/repos_real.jsonl"


# 数据预处理
def process(input_files=None, output_files=None):
    if input_files is None:
        input_files = [default_input_real]
    if output_files is None:
        output_files = [default_output_real]
    for input_file, output_file in zip(input_files, output_files):
        with jsonlines.open(input_file) as reader:
            with jsonlines.open(output_file, 'w') as writer:
                repo_dicts = list(reader)
                random.shuffle(repo_dicts)
                random.shuffle(repo_dicts)
                random.shuffle(repo_dicts)
                
                # 不同字段采取不同处理方式
                for repo_dict in repo_dicts:
                    repo_dict["name"] = preprocess.name_process(repo_dict["name"])
                    repo_dict["topics"] = preprocess.topics_process(repo_dict["topics"])

                    repo_dict["files"] = preprocess.files_process(repo_dict["files"])

                    repo_dict["description"] = preprocess.description_process(repo_dict["description"])
                    repo_dict["readme"] = preprocess.readme_process(repo_dict["readme"])
                    repo_dict["label"] = 0
                    writer.write(repo_dict)


if __name__ == '__main__':
    process()
    split_label_repos.real_repo()

