# pandas : 행과 열이 단순 정수형 인덱스가 아닌 레이블로 식별되는 numpy의 구조화된 배열을 보완(보강)하는 모듈
#          고수준의 자료구조(시계열 축약연산, 누락데이터 처리, SQL, 시각화...)
import pandas as pd
from pandas import Series
from pandas import DataFrame
import numpy as np

# Series : 일련의 객체를 담을 수 있는 1차원 배열과 유사한 자료구조로 색인을 가짐
# list, array로부터 만들 수 있음 
obj = Series([3, 7, -5, 4]) # list, tuple은 가능 -> 순서가 있기 때문    set은 불가능 -> 순서가 없기 때문
print(obj, type(obj)) # 자동 인덱스  (명시적)
print(obj[1])
obj2 = Series([3, 7, -5, 4], index=['a','b','c','d'])
print(obj2)
print(obj2['b'])
print(obj2.sum(), np.sum(obj2))
print(obj2.values)
print(obj2.index)
# 슬라이싱 
print(obj2['a'])  # 3
print(obj2[['a']]) # 인덱스와 값이 같이 나옴  a   3 
print(obj2[['a', 'b']])
print(obj2['a':'b'])
print(obj2[3])
print(obj2.iloc[3])
print(obj2[[2,3]])
print(obj2.iloc[[2,3]])
print(obj2 > 0)
print('a' in obj2) # obj2 에 a라는 애가 있는지 
print('\ndict type으로 Series 객체 생성')
names = {'mouse': 5000,'keyboard':25000, 'monitor':450000}
print(names, type(names))
obj3 = Series(names)
print(obj3, type(obj3))
obj3.index = ['마우스', '키보드', '모니터']
print(obj3)
print(obj3['마우스'])
print(obj3[0])

obj3.name = '상품가격'
print(obj3)

print('-------------------------------------')
# DataFrame : Series 객체가 모여 표를 구성
df = DataFrame(obj3)
print(df)

data = {  # dict type
    'irum':['홍길동','한국인','신기해','공기밥','한가해'],
    'juso':('역삼동','신당동','역삼동','역삼동','신사동'),
    'nai':[23, 25, 33, 30, 35]
}
frame = DataFrame(data)
print(frame)

print(frame['irum'])
print(frame.irum)
print(type(frame.irum))  # <class 'pandas.core.series.Series'>
print(DataFrame(data, columns=['juso', 'irum', 'nai']))

print('\ndata에 NaN을 넣기')
frame2 = DataFrame(data, columns=['irum','nai','juso','tel'],
                   index=['a','b','c','d','e'])
print(frame2)   
frame2['tel'] = '111-1111'
print(frame2)

val = Series(['222-2222','333-3333','444-4444'], index=['b','c','e'])
frame2.tel = val
print(frame2)

print(frame2.T)  # 양식(배열)을 바꿈 

print(frame2.values, type(frame2.values)) # <class 'numpy.ndarray'>   # 2차원 배열 형식으로 나옴(중복 리스트로 반환이 됨)
print(frame2.values[0,1])
print(frame2.values[0:2])

# 행 / 열 삭제
frame3 = frame2.drop('d', axis=0)  # 인덱스가 d인 행 삭제 # 'axis=0'은 안써줘도 상관 없음 
print(frame3)

frame4 = frame2.drop('tel', axis=1)  # 열 이름이 tel인 열이 삭제
print(frame4)

print('정렬----')
print(frame2.sort_index(axis=0, ascending=False))   # 행 단위로 내림차순 정렬  # True이면 오름차순 정렬 
print(frame2.sort_index(axis=1, ascending=True))  # 열단위 오름차순 정렬

print(frame2['juso'].value_counts())

print('문자열 자르기 ---------')
data = {
    'juso':['강남구 역삼동', '중구 신당동', '강남구 대치동'],
    'inwon':[23, 25, 15]
}
fr = pd.DataFrame(data)
print(fr)
result1 = Series([x.split()[0] for x in fr.juso])  # 공백을 구분자로 문자열 분리
result2 = Series([x.split()[1] for x in fr.juso])
print(result1, result1.value_counts())  # 건수를 세는 것 
print(result2, result2.value_counts())

