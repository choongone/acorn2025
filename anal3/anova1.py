# 세개 이상(요인은 하나)의 모집단에 대한 가설검정–분산분석
# ‘분산분석’ -> 분산이 발생한 과정을 분석하여 요인에 의한 분산과 요인을 통해 나누어진 각 집단 내의 분산으로 나누고 요인에 의한 분산이 의미있는 크기를 크기를 가지는지를 검정하는 것을 의미
# 세집단 이상의 평균비교에서는 독립인 두 집단의 평균비교를 반복하여 실시 할 경우에 제1종오류가 증가하게 되어 문제가 발생
# 이를 해결하기 위해 Fisher가 개발한분산분석(ANOVA, ANalysis Of Variance)을 이용
# 분산의 성질과 원리를 이용하여, 평균의 차이를 분석
# 즉, 평균을 직접 비교하지 않고, 집단 내 분산과 집단 간 분산을 이용해 집단의 평균이 서로 다른지 확인하는 방법임

# f-value = 그룹 간 분산(Between variance) / 그룹 내 분산(Within variance)
# 의미 없는 그룹 내 분산보단 그룹 간 분산이 커져야 f-value값이 증가 -> 유의미해짐 

# 서로 독립인 세집단의 평균차이 검정
# 실습) 세 가지 교육방법을 적용하여 1개월 동안 교육받은 교육생 80명을 대상으로 실기시험을 실시.   three_sample.csv
# 독립변수 = 교육방법(요인)  ->  요인은 하나이나 교육방법이 세가지 방법이 있음 
# 시험점수 = 종속변수
# 일원분산분석(one-way anova) - 복수의 집단을 대상으로 집단을 구분하는 요인이 하나인 분석

# 귀무 : 세 가지 교육방법을 통한 실기시험 평균에 차이가 없다. 
# 대립 : 세 가지 교육방법을 통한 실기시험 평균에 차이가 있다. 

import pandas as pd
import scipy.stats as stats
from statsmodels.formula.api import ols

data = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/three_sample.csv')
print(data.head(3))
print(data.shape)   # (80, 4)
print(data.describe())  # score에 의심 데이터 발견 

# 이상치를 차트로 확인
import matplotlib.pyplot as plt
# plt.hist(data.score)
# plt.boxplot(data.score)
# plt.show()
# plt.close()

# 이상치 제거
data = data.query('score <= 100')
print(len(data))    # 78

result = data[['method', 'score']]
print(result)
m1 = result[result['method'] == 1]
m2 = result[result['method'] == 2]
m3 = result[result['method'] == 3]
print(m1[:3])
print(m2[:3])
print(m3[:3])
score1 = m1['score']
score2 = m2['score']
score3 = m3['score']


# 정규성 
print('score1 : ', stats.shapiro(score1).pvalue)    # 0.174673  > 0.05
print('score2 : ', stats.shapiro(score2).pvalue)    # 0.33190   > 0.05
print('score3 : ', stats.shapiro(score3).pvalue)    # 0.115585  > 0.05
print(stats.ks_2samp(score1, score2))   # 두 집단의 동일분포 여부 확인 

# 등분산성(복수 집단 분산의 치우침 정도)
print('levene:', stats.levene(score1, score2, score3).pvalue)
print('flinger:', stats.fligner(score1, score2, score3).pvalue)
print('bartlett:', stats.bartlett(score1, score2, score3).pvalue)

print('---------------')
# 교차표 등 작성가능 ...

import statsmodels.api as sm
reg = ols("data['score'] ~ C(data['method'])", data=data).fit()     # 단일회귀 모델 작성    
        # 종속변수 먼저 적음

# 분산 분석표를 이용해 분산결과 작성         
table = sm.stats.anova_lm(reg, type=2)
print(table)    # pvalue = 0.939639 > 0.05이므로 귀무채택 

# 사후 검정(Post Hoc Test)
# 분산분석은 집단의 평균에 차이 여부만 알려줄 뿐, 각 집단 간의 평균 차이는 알려주지 않음
# 각 집단 간의 평균 차이를 확인하기 위해 사후검정 실시 

from statsmodels.stats.multicomp import pairwise_tukeyhsd
turResult = pairwise_tukeyhsd(endog=data.score, groups=data.method)
print(turResult)
turResult.plot_simultaneous(xlabel='mean', ylabel='group')
plt.show()
plt.close()





















