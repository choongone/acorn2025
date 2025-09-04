# XML로 제공되는 날씨자료 처리 

import urllib.request
import urllib.parse
from bs4 import BeautifulSoup
import pandas as pd

url ="http://www.kma.go.kr/XML/weather/sfc_web_map.xml"
# data = urllib.request.urlopen(url).read()
# print(data.decode('utf8'))
soup = BeautifulSoup(urllib.request.urlopen(url), 'xml')
# print(soup)

local = soup.find_all('local')

data = []
for loc in local:
    city = loc.text
    temp = loc.get('ta')
    data.append([city, temp])

df = pd.DataFrame(data, columns=['지역', '온도'])
# print(df.head(3))
df.to_csv('weather.csv', index=False)


df = pd.read_csv('weather.csv')
print(df.head(2))
print(df[0:2])
print(df.tail(2))
print(df[-2:len(df)])
print()
print(df.iloc[0:2, :])
print()
print(df.loc[1:3],['온도'])
print(df.info())
print(df['온도'].mean())
print(df['온도'] >= 30)
print(df.loc[df['온도'] >= 32])
print(df.sort_values(['온도'], ascending = True))
