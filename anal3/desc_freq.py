# 기술 통계의 목적: 데이터를 수집, 요약, 정리, 시각화 
# 도수분포표(Frequency Distribution Table): 데이터를 구간별로 나눠 빈도를 정리한 표
# 이를 통해 데이터의 분포를 한 눈에 파악할 수 있음

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rc('font', family='malgun gothic')
plt.rcParams['axes.unicode_minus'] = False

# step1: 데이터를 읽어 DataFrame에 저장
df = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/heightdata.csv')
print(df.head(2))

# step2: max, min 
min_height = df['키'].min()
max_height = df['키'].max()
print(f'최소값:{min_height},최대값:{max_height}')  # 최소값: 158, 최대값: 191

# step3: 계급 설정 (cut)
bins = np.arange(156, 195, 5)
print(bins)
df['계급'] = pd.cut(df['키'], bins=bins, right=True, include_lowest=True)  # 구간의 오른쪽이 포함
print(df.head(3))  # 자료에서 앞에서 3개만 추출
print(df.tail(3))  # 자료에서 뒤에서 3개만 추출

# step4: 각 계급의 중앙값  (156 + 161)/2
df['계급값'] = df['계급'].apply(lambda x:int((x.left + x.right) / 2))
print(df.head(3))

# step5: 도수 계산
freq = df['계급'].value_counts().sort_index()

# step6: 상대도수(전체 데이터에 대한 비율) 계산
relative_freq = (freq / freq.sum()).round(2)
print(relative_freq)

# step7: 누적 도수 계산 - 계급별 도수를 누적 
cum_freq = freq.cumsum()

# step8: 도수분포표 작성
# "156 ~ 161" 이런 모양 출력하기
dist_table = pd.DataFrame({
    '계급':[f"{int(interval.left)} ~ int(interval.right)"  for interval in freq.index],   # 모양 꾸미기
# 계급의 중간값
    '계급값':[int((interval.left + interval.right) / 2)  for interval in freq.index],  
    '도수':freq.values,
    '상대도수':relative_freq.values, 
    '누적도수':cum_freq.values
})
print('도수분포표')
print(dist_table.head(3))

# step9: 히스토그램 그리기 
plt.figure(figsize=(8, 5))
plt.bar(dist_table['계급값'], dist_table['도수'], width=5, color='cornflowerblue', edgecolor='black')
plt.title('학생 50명 키 히스토그램', fontsize=16)
plt.xlabel('키(계급값)')
plt.ylabel('도수')
plt.xticks(dist_table['계급값'])  # 간격 
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()




