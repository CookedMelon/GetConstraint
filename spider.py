# pre标签内的直接省去

# 爬虫获取网页源代码

import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

# 创建 Chrome 无头浏览器选项
chrome_options = Options()
chrome_options.add_argument('--headless')  # 启用无头模式

# 创建 Chrome 无头浏览器实例
driver = webdriver.Chrome(options=chrome_options)
api_list=[]

# print(api_list)
def load_api_list2(apis):
    global api_list
    for api in apis:
        api=api.replace('.','/')
        api='https://www.tensorflow.org/api_docs/python/'+api
        api_list.append(api)
def load_api_list(api_path):
    global api_list
    with open(api_path,'r') as f:
        for line in f.readlines():
            # 替换其中的.为/
            line=line.replace('.','/')
            line='https://www.tensorflow.org/api_docs/python/'+line
            api_list.append(line.strip())
def spider(url):


    # 执行自动化操作
    driver.get('https://example.com')  # 打开网页
    # 在这里执行你的网页操作

    # 最后关闭浏览器
    driver.quit()



def main():
    # api_path='/home/cc/Workspace/tfconstraint/tf_valid_apis.txt'
    apis=['tf.data.experimental.assert_cardinality']
    load_api_list2(apis)
    for url in api_list:
        print('url',url)
        spider(url)
    # for url in api_list:
    #     spider(url)
if __name__ == '__main__':
    main()