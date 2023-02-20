import long_text
import short_text


# process for short text
def name_process(name: str):
    return short_text.process(name)


def topics_process(topics: list):
    return_list = []
    for topic in topics:
        return_list.append(short_text.process(topic))
    return return_list


def files_process(files: list):
    return_list = []
    for file in files:
        return_list.append(short_text.process(file))
    return return_list


# process for long text
def description_process(descr: str):
    return long_text.process(descr)


def readme_process(readme: str):
    return long_text.process(readme)
