# 추론통계 분석 중 비율검정
# -비율검정특징
# : 집단의 비율이 어떤 특정한 값과 같은지를 검증.
# : 비율 차이 검정통계량을 바탕으로 귀무가설의 기각 여부를 결정.

# one-sample
# A회사에는100명 중에45명이 흡연을한다.국가통계를 보니 국민흡연율은 35%라고 한다. 비율이같냐?
# 귀무 : A회사 직원들의 흡연율과 국민 흡연율의 비율이 같다. 
# 대립 : A회사 직원들의 흡연율과 국민 흡연율의 비율이 같지 않다. 
import numpy as np 
from statsmodels.stats.proportion import proportions_ztest
count = np.array([45])
nobs = np.array([100])
val = 0.35
z, p = proportions_ztest(count=count, nobs=nobs, value=val)
print(z)    # [2.01007563]
print(p)    # [0.04442318] < 0.05   귀무기각: 비율이 다르다. 

'------------------------------------------------'

# two-sample
#  A회사사람들300명 중100명이 커피를마시고,B회사사람들400명중170명이커피를마셨다. 비율이같냐?
count=np.array([100, 170])
nobs = np.array([300, 400])
z, p = proportions_ztest(count=count, nobs=nobs, value=val)
print(z)    # -11.88004694268173
print(p)    # 1.5028294294082938e-32 < 0.05   귀무기각: 비율이 다르다. 

print('---------------이항 검정---------------')
# '결과가 두 가지 값을 가지는 확률변수의 분포를 판단'하는데 효과적 
# ex) 10명의 자격증 시험 합격자 중 여성이 6명이었다고 할 때, '여성이 남성보다 합격률이 높다'라고 할 수 있는가?
import pandas as pd 
import scipy.stats as stats

data = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/one_sample.csv')
print(data.head(3))
ctab = pd.crosstab(index=data['survey'], columns='count')
ctab.index = ['불만족', '만족']
print(ctab)
# col_0    count
# 불만족      14
# 만족       136

# 귀무 : 직원 대상으로 고객 대응 교육 후 고객 안내 서비스 만족률이 80%이다. 
# 대립 : 직원 대상으로 고객 대응 교육 후 고객 안내 서비스 만족률이 80%가 아니다. 

# 양측 검정 : 방향성이 없다. 
# x = stats.binomtest([136, 14], p=0.8, alternative='two-sided')  ---->  이제 이거 안됨

# n=150번 시행 중 성공 
result = stats.binomtest(k=136, n=150, p=0.8, alternative='two-sided')
print(result.pvalue)    # 0.0006734701362867024 < 0.05  귀무기각 

# 단측 검정 : 방향성이 있다.    (80%보다 크다고 가정하고 검증)
result = stats.binomtest(k=136, n=150, p=0.8, alternative='greater')   # less : 작다.     greater: 크다. 
print(result.pvalue)    # 0.00031794019219854805 < 0.05 귀무기각









