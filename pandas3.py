from pandas import Series, DataFrame
import numpy as np

s1 = Series([1,2,3], index=['a', 'b', 'c'])
s2 = Series([4,5,6,7], index=['a', 'b', 'd', 'c'])
print(s1)
print(s2)
print(s1 + s2)
print(s1.add(s2))
print(s1.multiply(s2))  # sub, div
print()
df1 = DataFrame(np.arange(9).reshape(3,3), columns=list('kbs'), index=['서울', '대전', '대구'])
df2 = DataFrame(np.arange(12).reshape(4,3), columns=list('kbs'), index=['서울', '대전', '제주', '수원'])
print(df1)
print(df2)
print(df1 + df2)  # NaN
print(df1.add(df2, fill_value=0))  # NaN은 0으로 채운 후 연산에 참여 
print()

ser1 = df1.iloc[0]
print(ser1)
print(df1 - ser1)  # Broadcasting 연산

print('---결측치, 기술적 통계 관련 함수---')
# 결측치 
df = DataFrame([[1.4, np.nan],[7, -4.5],[np.nan, None], [0.5, -1]], columns=['one', 'two'])
print(df)
print(df.isnull())  # null 값을 확인  # null인 부분은 True
print(df.notnull())
print(df.drop(0))  # 특정 행 삭제 (NaN과 관계 X)
print(df.dropna())  # NaN 값이 포함된 모든 행 삭제
print(df.dropna(how='any'))  # 위와 같은 의미 # any가 디폴트 값 
print(df.dropna(how='all')) 
print(df.dropna(subset=['one']))   # any, all, subset 셋이 차이가 있음 
print(df.dropna(axis='rows'))
print(df.dropna(axis='columns'))
print(df.fillna(0))


print('---기술 통계---')
print(df.sum())  # 열의 합 - NaN은 연산X
print(df.sum(axis=0))
print(df.sum(axis=1))
print(df.describe())  # 요약 통계량을 보여줌
print(df.info())  # 구조 확인

print('---재구조화, 구간 설정, agg 함수---')
df = DataFrame(1000 + np.arange(6).reshape(2, 3), index=['서울', '대전'], columns=['2020', '2021', '2022'])
print(df)
print(df.T)
# stack, unstack
df_row = df.stack()  # 열 -> 행으로 변경 , 열 쌓기 
print(df_row)

df_col = df_row.unstack()  # 행 -> 열로 복원 
print(df_col)

print()
import pandas as pd
# 구간 설정 : 연속형 자료를 범주화 
price = [10.3, 5.5, 7.8, 3.6]  
cut = [3, 7, 9, 11]  # 구간 기준값 
result_cut = pd.cut(price, cut)
print(result_cut)  # (): 초과 []: 이하
print(pd.value_counts(result_cut))

datas = pd.Series(np.arange(1, 1001))
print(datas.head(3))
print(datas.tail(3))
result_cut2 = pd.qcut(datas, 3)
print(result_cut2)
print(pd.value_counts(result_cut2))

print('-----------')
group_col = datas.groupby(result_cut2)
# print(group_col)
print(group_col.agg(['count','mean','std','min']))

def myFunc(gr):
    return{
        'count':gr.count(),
        'mean':gr.mean(),
        'std':gr.std(),
        'min':gr.min()
        }
print(group_col.apply(myFunc))
print(group_col.apply(myFunc).unstack())
