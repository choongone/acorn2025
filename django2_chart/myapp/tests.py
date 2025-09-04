import numpy as np
from bs4 import BeautifulSoup
import urllib 
import requests

# try:
#       url = "http://www.naver.com"
#       page = urllib.request.urlopen(url)
           
#       soup = BeautifulSoup(page.read(), "lxml") 
#       title = soup.find('ol'). find_all('li')
#       for i in range(0, 10):
#               print(str(i + 1) + ") " + title[i].a['title'])
# except Exception as e:
#         print('에러:', e)

# df = DataFrame(np.arange(12).reshape(4, 3), index=['1월','2월','3월','4월'],
#                columns=['강남','강북','서초'])
# print(df)

# from pandas import DataFrame
# frame = DataFrame({'bun':[1,2,3,4], 'irum':['aa','bb','cc','dd']}, index=['a','b', 'c','d'])
# print(frame.T)
# frame2 =frame.drop('d', axis=0)     # 인덱스가 'd'인 행 삭제
# print(frame2)


import pandas as pd
from pandas import DataFrame
# df=pd.DataFrame({'data':range(8), 'key':['b','b','b','c','a','a','b']})
# print('df1:\n', df)


# x = np.array([1,2,3,4,5])
# y = np.array([1,2,3]).reshape(3,1)
# print(x+y)

import pandas as pd
import numpy as np

df = pd.DataFrame(np.random.randn(9, 4))
df.columns = ['가격1', '가격2', '가격3', '가격4']
print(df)
print()
print(df.mean())