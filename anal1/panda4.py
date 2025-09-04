import pandas as pd
from pandas import Series, DataFrame
import numpy as np

df1 = pd.DataFrame({'data1':range(7), 'key':['b','b','b','c','a','a','b']})
print('df1:\n', df1)
df2 = pd.DataFrame({'key':['a','b','d'], 'data2':range(3)})
print('df2:\n', df2)
print()

# merge 함수
print('merge:\n', pd.merge(df1, df2, on='key'))  # key를 기준으로 병합(inner join, 교집합)
print()

# merge 함수의 how 속성
print('merge how=inner:\n', pd.merge(df1, df2, on='key', how='inner'))  # inner join (교집합)
print('merge how=outer:\n', pd.merge(df1, df2, on='key', how='outer'))  # outer join (합집합). 결과에 없는 값은 NaN. full outer join
print('merge how=left:\n', pd.merge(df1, df2, on='key', how='left'))  # left join (왼쪽 DataFrame 기준)
print('merge how=right:\n', pd.merge(df1, df2, on='key', how='right'))  # right join (오른쪽 DataFrame 기준)

print('--' * 30)
# 공통 column이 없는 경우
df3 = pd.DataFrame({'key2':['a','b','c'], 'data2':range(3)})
print('df1:\n', df1)
print('df3:\n', df3)
print('merge left_on:\n', pd.merge(df1, df3, left_on='key', right_on='key2'))  # left_on과 right_on을 사용하여 병합. inner join

print('--' * 30)
# concat 함수
print('concat axis=0:\n', pd.concat([df1, df3], axis=0))  # 세로 방향으로 연결. 기본값은 axis=0
print('concat axis=1:\n', pd.concat([df1, df3], axis=1))  # 가로 방향으로 연결

print()
s1= pd.Series([0,1], index=['a','b'])
s2= pd.Series([2,3,4], index=['c','d','e'])
s3= pd.Series([5,6], index=['f','g'])
print('concat Series:\n', pd.concat([s1, s2, s3]))  # Series 연결

# 그룹화: pivot_table
data = {'city': ['강남', '강북', '강남', '강북'],
        'year': [2000, 2001, 2002, 2003],
        'population': [3.3, 2.5, 3.0, 2.0]}
df = pd.DataFrame(data)
print('df:\n', df)
print('pivot_table:\n', df.pivot_table(index='city', columns='year', values='population'))
print()
print(df.set_index(['city', 'year']).unstack())  # pivot_table과 동일한 결과
print('describe:\n', df.describe())  # describe 함수로 통계 요약
# pivot_table : pivot과 groupby의 조합. 데이터 집계와 재구성에 유용
print('city기준 인구수 평균:\n',df.pivot_table(index='city'))  # city를 기준으로 집계. 기본적으로 mean() 함수 사용(평균값)
print('city기준 길이, 합계 계산:\n', df.pivot_table(index='city', aggfunc=[len,sum]))  # city를 기준으로 집계. len()과 sum() 함수 사용
print('city기준 인구수 평균, 합계 계산:\n', df.pivot_table(index='city', values='population', aggfunc=[len,sum]))  # city를 기준으로 population의 평균값과 합계 계산
print('city기준 인구수, index를 year로 설정, 빈값은 0:\n',
       df.pivot_table(index='city', columns='year', values='population', margins=True, fill_value=0))  
        # city를 기준으로 population의 평균값을 year별로 집계. 빈값은 0으로 채움. margins=True는 전체 합계 추가
print()
hap = df.groupby(['city'])
print('hap:\n', hap)  # city를 기준으로 그룹화
print('hap.sum():\n', hap.sum())  # city를 기준으로 합계값
print('groupby사용:\n', df.groupby(['city']).sum())  # groupby를 사용한 합계값
print('groupby 사용. city, year 평균\n', df.groupby(['city', 'year']).mean())  # city와 year를 기준으로 평균값

#file i/o
df = pd.read_csv('anal2/ex1.csv') # df = pd.read_csv('anal2/ex1.csv', sep=',')
print('ex1.csv 읽기:\n', df, type(df))
print('ex1.csv info:\n', df.info())

df = pd.read_table('anal2/ex1.csv', sep=',')  # read_table은 기본적으로 tab으로 구분된 파일을 읽음. csv파일을 읽을땐 sep를 반드시 지정해야 함
print('ex1.csv table로 읽기:\n', df, type(df))
print('--' * 30)
df = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/ex2.csv')  # URL에서 CSV 파일 읽기
print('ex2.csv 읽기:\n', df)

print('--' * 30)
df = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/ex2.csv', header=None)  # header를 None으로 지정하면 첫 번째 행을 데이터로 읽음
print('ex2.csv header=None으로 읽기:\n', df)
print('--' * 30)
df = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/ex2.csv', 
                 header=None, names=['a', 'b'])  # header를 None으로 지정하고, names로 컬럼 이름 지정
print('ex2.csv header=None, names 지정으로 읽기:\n', df) # names로 컬럼 이름 지정시 오른쪽부터 채워짐
df = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/ex2.csv', 
                 header=None, names=['a', 'b', 'c', 'd', 'msg'], skiprows=1)  # header를 None으로 지정하고, names로 컬럼 이름 지정, skiprows로 첫 번째 행 건너뜀
print('ex2.csv header=None, names 지정으로 읽기(모두 채움), 첫번째 행 건너뜀:\n', df)

print('--' * 30)
df = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/ex3.txt') # txt 파일 읽기
print('ex3.txt 읽기:\n', df)
print('ex3.txt info:\n', df.info())
print('--' * 30)
df = pd.read_table('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/ex3.txt', sep='\s+')  # 공백으로 구분된 txt 파일 읽기 (정규표현식)
print('ex3.txt 공백으로 구분된 파일 읽기:\n', df)
print('info:\n', df.info())

print('--' * 30)
df = pd.read_fwf('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/data_fwt.txt', 
                 widths=[10, 3, 5], names=('data', 'name', 'price'), encoding='utf-8')  # 고정폭 파일 읽기. widths로 각 칼럼의 너비 지정(글자수). names로 칼럼 이름 지정
print('고정폭 파일 읽기:\n', df)
print()
url = "https://ko.wikipedia.org/wiki/%EB%A6%AC%EB%88%85%EC%8A%A4"
df = pd.read_html(url)  # 웹 페이지에서 테이블 읽기
print('웹 페이지에서 테이블 읽기:\n', df)  # df는 리스트 형태로 반환됨. 각 테이블이 DataFrame으로 저장됨
print(f'총 {len(df)}개의 자료.')

# 대량의 데이터 파일을 읽는 경우 chunk 단위로 분리해 읽기 가능
# 1) 메모리 절약
# 2) 스트리밍 방식으로 순차적으로 처리 가능 (로그 분석)
# 3) 분산 처리(batch processing)
# 4) 다소 속도는 느릴 수 있음

import time
import matplotlib.pyplot as plt
plt.rc('font', family='Malgun Gothic')  # 한글 폰트 설정

n_rows = 10000
data = {
    'id': range(1, n_rows + 1),
    'name': [f'Student_{i}' for i in range(1, n_rows + 1)],
    'score1': np.random.randint(0, 101, size=n_rows),
    'score2': np.random.randint(0, 101, size=n_rows)
}
df = pd.DataFrame(data)
print('df head:\n', df.head(3))
print('df tail:\n', df.tail(3))

csv_path = 'anal2/students.csv'
df.to_csv(csv_path, index=False)  # DataFrame을 CSV 파일로 저장
