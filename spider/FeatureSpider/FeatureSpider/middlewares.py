import random


class RandomHeadersMiddleware(object):
    tokens = [
        # 陶望成
        # "token " + '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',
        # "token " + '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',
        # # 王卓
        # "token " + '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',
        # "token " + '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',
        # "token " + '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',
        # "token " + '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',
        # "token " + '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',
        # "token " + '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',
        # "token " + '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',
        # "token " + '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',
        # # 陈旭洋
        # "token " + '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',
        # "token " + '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',
        # "token " + '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',
        # "token " + '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',
        # "token " + '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',
        # "token " + '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',
        # "token " + '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',
        # "token " + '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',
        # # 张明昊
        # "token " + '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',
        # "token " + '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',
        # "token " + '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',
        # "token " + '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',
        # "token " + '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',
        # # 吴良轩
        # "token " + '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',
        # "token " + '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',
        # "token " + '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',
        # "token " + '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',
        # "token " + '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',
        # # 李洋瑞
        # "token " + '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',
        #"token " + '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'


        "token " + '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',
        "token " + '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',
        
        "token " + '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',
        "token " + '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',
        "token " + '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',
        "token " + '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',
        "token " + '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',
        "token " + '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',
        "token " + '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',
        "token " + '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',
        

        "token " + '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',

    ]
    tokens_len = len(tokens)
    user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_8_3) AppleWebKit/536.5 (KHTML, like Gecko) ' \
                 'Chrome/19.0.1084.54 Safari/536.5'
    choice = random.choice(range(tokens_len))

    def process_request(self, request, spider):
        request.headers['Authorization'] = self.tokens[self.choice]
        request.headers['User-Agent'] = self.user_agent
        self.choice = (self.choice + 1) % self.tokens_len
