# 온도(세 개의 집단)에 따른 음시점 매출액의 평균차이 검정 
# 공통 칼럼이 년월일인 두 개의 파일을 조합을 해서 작업 

# 귀무 : 온도에 따른 음식점 매출액의 평균차이는 없다. 
# 대립 : 온도에 따른 음식점 매출액의 평균차이는 있다. 
import numpy as np
import pandas as pd
import scipy.stats as stats

# 매출 자료 읽기 
sales_data = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/tsales.csv', dtype={'YMD':'object'})
print(sales_data.head(3))   # 328 rows, 3 columns   YMD:20190514
print(sales_data.info())

# 날씨 자료 읽기
wt_data = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/tweather.csv')
print(wt_data.head(3))      # 702 rows, 9 columns   tm:2018-06-01
print(wt_data.info())

# sales 데이터의 날짜를 기준으로 두 개의 자료를 병합 작업 진행 
# wt_data.tm = wt_data.tm.map(lambda x:x.replace)을 아래처럼 수정
wt_data['tm'] = wt_data['tm'].str.replace('-', '')
print(wt_data.head(3))

frame = sales_data.merge(wt_data, how='left', left_on='YMD', right_on='tm')
print(frame.head(3), ' ', len(frame))
# 'YMD', 'AMT', 'CNT', 'stnId', 'tm', 'avgTa', 'minTa', 'maxTa', 'sumRn', 'maxWs', 'avgWs', 'ddMes'
data = frame.iloc[:, [0,1,7,8]]     # 날짜, 매출액, 최고기온, 강수량
print(data.head(3))

print(data.maxTa.describe())
# 일별 최고온도(연속형) 변수를 이용해 명목형(구간화) 변수 추가
data['ta_gubun'] = pd.cut(data.maxTa, bins=[-5, 8, 24, 37], labels=[0, 1, 2])     # 세 구간으로 나눔
print(data.head(3))
print(data.ta_gubun.unique())   # [2, 1, 0]
print(data.isnull().sum())

# 최고온도를 세 그룹으로 나눈 뒤, 등분산/정규성 검정
x1 = np.array(data[data.ta_gubun == 0].AMT)
x2 = np.array(data[data.ta_gubun == 1].AMT)
x3 = np.array(data[data.ta_gubun == 2].AMT)
print(x1[:5], len(x1))
print()
print(stats.levene(x1, x2, x3).pvalue)  # 0.0390 등분산 만족 X
print(stats.shapiro(x1).pvalue)     # 정규성은 어느정도 만족 
print(stats.shapiro(x2).pvalue)
print(stats.shapiro(x3).pvalue)

spp = data.loc[:, ['AMT', 'ta_gubun']]
print(spp.groupby('ta_gubun').mean())
print(pd.pivot_table(spp, index=['ta_gubun'], aggfunc='mean'))

# ----------------------------------------------------------------- #

# ANOVA 진행 

sp = np.array(spp)
group1 = sp[sp[:, 1]==0, 0]
group2 = sp[sp[:, 1]==1, 0]
group3 = sp[sp[:, 1]==2, 0]

print(stats.f_oneway(group1, group2, group3))   # pvalue=2.360737101089604e-34 < 0.05  귀무기각 -> 온도에 따라 매출액 차이가 있다. 

# 참고 : 등분산성 만족 X : Welch's anova test 사용
# pip install pingouin
from pingouin import welch_anova
print('welch_anova: ' ,welch_anova(dv='AMT', between='ta_gubun', data=data))     # pvalue=7.907874e-35 < 0.05 귀무기각 


# 참고 : 정규성 만족 X : Kruskal Wallis test 사용
print('kruskal : ', stats.kruskal(group1, group2, group3))      # pvalue=1.5278142583114522e-29 < 0.05 귀무기각 
# 온도에 따라 매출액의 차이가 유의미함 

# 사후분석 
from statsmodels.stats.multicomp import pairwise_tukeyhsd
turkeyResult = pairwise_tukeyhsd(endog=spp['AMT'], groups=spp['ta_gubun'])
print(turkeyResult)

import matplotlib.pyplot as plt
turkeyResult.plot_simultaneous(xlabel='mean', ylabel='group')
plt.show()
plt.close()

















