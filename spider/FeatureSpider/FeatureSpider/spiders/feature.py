import scrapy
from FeatureSpider.items import FeaturespiderItem
import json
import base64


class FeatureSpider(scrapy.Spider):
    name = 'feature'
    allowed_domains = ['github.com']

    def start_requests(self):
        # fp = open('/home/ubuntu/code/python/demo/repos_list.json', "r")
        # repos_dict = json.load(fp)
        # fp.close()

        fp = open(r'D:\JetBrains\Python\WorkSpace\demo\repos_list.txt', "r")
        repos = fp.readlines()
        fp.close()

        pre_url = 'https://api.github.com/repos/'
        keywords = []
        for repo in repos:
            try:
                repo_url = pre_url + repo.strip()
            except:
                print(repo)
                continue
            yield scrapy.Request(url=repo_url, callback=self.parse,
                                 cb_kwargs=dict(keywords=keywords), dont_filter=True)

    def parse(self, response, keywords):
        item = FeaturespiderItem()
        item['repo_dict'] = {}
        repo_dict = response.json()
        item['repo_dict']['url'] = repo_dict['html_url']
        item['repo_dict']['api_url'] = repo_dict['url']
        item['repo_dict']['full_name'] = repo_dict['full_name']
        item['repo_dict']['name'] = repo_dict['name']
        item['repo_dict']['topics'] = repo_dict['topics']
        item['repo_dict']['files'] = []
        item['repo_dict']['readme'] = ''
        item['repo_dict']['description'] = repo_dict['description']

        # repo_dict['readme'] = ''
        # repo_dict['files'] = []
        # repo_dict['keywords'] = keywords
        # item['repo_dict'] = repo_dict


        contents_url = repo_dict['contents_url'].split('/')
        contents_url.pop(-1)
        dir_urls = ['/'.join(contents_url)]

        yield scrapy.Request(dir_urls[0], callback=self.readme_file,
                             cb_kwargs=dict(item=item, dir_urls=dir_urls), dont_filter=True)

    def readme_file(self, response, item, dir_urls):
        contents = response.json()

        if isinstance(contents, list):
            readme_files = []
            for content in contents:
                if content['name'].startswith('README'):
                    if content['name'] == 'README.md':
                        readme_files.insert(0, content)
                    else:
                        readme_files.append(content)

            if len(readme_files) == 0:
                yield scrapy.Request(dir_urls[0], callback=self.file_finder,
                                     cb_kwargs=dict(item=item, dir_urls=dir_urls), dont_filter=True)
            elif len(readme_files) >= 1:
                yield scrapy.Request(readme_files[0]['url'], callback=self.readme_file_process,
                                     cb_kwargs=dict(item=item, dir_urls=dir_urls), dont_filter=True)
        else:
            print('This is an empty repo', dir_urls[0])

    def readme_file_process(self, response, item, dir_urls):
        item['repo_dict']['readme'] = base64.b64decode(response.json()['content'].encode()).decode()

        yield scrapy.Request(dir_urls[0], callback=self.file_finder,
                             cb_kwargs=dict(item=item, dir_urls=dir_urls), dont_filter=True)

    def file_finder(self, response, item, dir_urls):
        contents = response.json()
        dir_urls.pop(0)
        for content in contents:
            if content['type'] == 'file':
                item['repo_dict']['files'].append(content['name'].split('.')[0])
            elif content['type'] == 'dir':
                dir_urls.append(content['url'])
                item['repo_dict']['files'].append(content['name'])
        if len(dir_urls) > 0:

            yield scrapy.Request(dir_urls[0], callback=self.file_finder,
                                 cb_kwargs=dict(item=item, dir_urls=dir_urls), dont_filter=True)
        else:
            yield item
        
