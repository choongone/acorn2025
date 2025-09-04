# 웹문서 읽기 
import urllib.request as req
from bs4 import BeautifulSoup
import urllib 

# 위키백과 문서 읽기 - 이순신 자료 
"""
url = "https://ko.wikipedia.org/wiki/%EC%9D%B4%EC%88%9C%EC%8B%A0"
wiki = req.urlopen(url)
print(wiki)  # 객체가 만들어짐
"""
"""
soup = BeautifulSoup(wiki, 'html.parser')
print(soup.select("body > div.mw-page-container > div > div.vector-column-start > p"))
"""


# 네이버 제공 코스피 정보 읽기 - DataFrame에 담아 어쩌구 저쩌구...
url_template = "https://finance.naver.com/sise/nxt_sise_market_sum.naver?&page={}"
csv_fname = '네이버코스피.csv'

import csv
import re
import pandas as pd
import requests
"""
with open(csv_fname, mode='w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f)
    # 제목
    headers = 'N 종목명	현재가	전일비	등락률	액면가	시가총액	상장주식수	외국인비율	거래량	PER	ROE'.split(' ')
    writer.writerow(headers)

    # 페이지 반복 
    for page in range(1, 3):
        url = url_template.format(page)
        res = requests.get(url)
        res.raise_for_status()  # 실패하면 작업 중지 
        soup = BeautifulSoup(res.text, 'html.parser')
        #rows = soup.find('table', attrs={'class':type_2}).find('tbody').find('tbody').find_all('tr')
        rows = soup.select('table.type_2 > tbody > tr')

    for row in rows:
        cols = row.find_all('td') # 각 행의 모든 열(td) 선택 
        if len(cols) <= len(headers):  
            print('[스킵됨] 열 수 부족 : |n', len(cols))
            continue
        row_data = [re.sub(r'[|n|t]+','',col.get_text(strip=True))for col in cols] # 각 열의 텍스트를 정리 
        writer.writerow(row_data)  # 각 행의 데이터를 CSV 파일에 작성            
print(f'{csv_fname} 저장 완료 ')
"""

df = pd.read_csv(csv_fname, dtype=str, index_col=False)
print(df.head(3))
print(df.columns.tolist())  # 컬럼명만 보기 
print(df.info())
numeric_cols = ['현재가', '전일비', '등락률', '액면가', '시가총액', '상장주식수', '외국인비율', '거래량', 'PER', 'ROE']

# 전일비 전용 전처리 함수 
def clean_change_direction(val):
    if pd.isna(val):
        return pd.NA
    val = str(val)
    val = val.replace(',','').replace('상승','+').replace('하락','-')
    val = re.sub(r'[^\d\.\-\+]', '', val)   # 숫자, -, + 등의 기호 외 문자들을 제거 
    try:
        return float(val)
    except ValueError:
        return pd.NA
    
df['전일비'] = df['전일비'].apply(clean_change_direction)
print(df.head(3))

# 일반 숫자형 컬럼 전처리 
def clean_numeric_column(seires):
    return(
        series.astype(str)
            .str.replace(',','', regex=False)
            .str.replace('%','', regex=False)
            .str.replace(['','-','N/A', 'nan'], pd.NA)
            .apply(lambda x:pd.to_numeric(x, errors='coerce')) 
    )

for col in numeric_cols:
    df[col] = clean_numeric_column(df[col])
print('숫자 컬럼 일괄 처리 후')
print(df.head(2))

print('--------')
print(df.describe())
print(df[['종목명','현재가','전일비']].head())
print('시가 총액 top 5')
top5 = df.dropna(subset=['시가총액']).sort_values(by='시가총액', ascending=False).head()
print(top5[['종목명', '시가총액']])

