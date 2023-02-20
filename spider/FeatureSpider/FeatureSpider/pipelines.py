# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html


# useful for handling different item types with a single interface
from itemadapter import ItemAdapter
import jsonlines


class FeaturespiderPipeline:
    def __init__(self):
        self.file = open(r"D:\JetBrains\Python\WorkSpace\demo\data\real\raw\repos_feature.jsonl", "w", encoding='utf-8')
        self.writer = jsonlines.Writer(self.file)

    def close_spider(self, spider):
        print('--------------关闭文件--------------')
        self.writer.close()
        self.file.close()

    def process_item(self, item, spider):
        self.writer.write(item['repo_dict'])
