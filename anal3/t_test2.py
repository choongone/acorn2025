# 독립 표본 검정 : 두 집단의 평균의 차이 검정
# 서로 다른 두 집단의 평균에 대한 통계 검정에 주로 사용
# 비교를 위해 평균과 표준편차 통계량을 사용
# 평균값의 차이가 얼마인지, 표준편차는 얼마나 다른지 확인해 분석 대상인 두 자료가 같을 가능성이 우연의 범위 5%(0.05)에 들어가는지를 판별
# ***결국 t-test는 두 집단의 평균과 표준편차 비율에 대한 대조 검정법*** 
# t-value = 차이/불확실성 -> 두 집단의 평균 차이가 커지면 t값은 커지고, p(불확실성)값은 작아짐   
# t와 p는 반비례함

# 서로 독립인 두 집단의 평균차이 검정(independentsamplest-test)
# 남녀의 성적, A반과 B반의 키, 경기도와 충청도의 소득 따위의 서로 독립인 두 집단에서 얻은 표본을 독립표본(twosample)이라고 한다.

from scipy import stats
import pandas as pd
import numpy as np

# 실습1 )남녀 두 집단 간 파이썬 시험의 평균차이검정
# 귀무 : 남녀 두 집단 간 파이썬 시험의 평균의 차이는 없다. 
# 대립 : 남녀 두 집단 간 파이썬 시험의 평균의 차이는 있다.
# 95% 신뢰수준에서 우연히 발생할 확률이 5%보다 작은가?
# 선행 조건 : 두 집단의 자료는 정규분포를 따른다. 분산이 동일하다.(등분산성)


male = [75, 85, 100, 72.5, 86.5] 
female = [63.2, 76, 52, 100, 70] 
print(np.mean(male), '  ', np.mean(female))      # 83.8 | 72.24

# two_sample = stats.ttest_ind(male, female)
two_sample = stats.ttest_ind(male, female, equal_var=True)      # 정규분포 분산은 같다고 가정

print(two_sample)       # TtestResult(statistic=1.233193, pvalue=0.25250768, df=8.0
# 해석 : pvalue=0.25250768 > 0.05  귀무채택 
# 남녀 두 집단 간 파이썬 시험의 평균의 차이는 없다. 두 집단 평균 차이가 유의하지 않다.(의미 있는 차이가 X)

print('\n--- 등분산 검정 ---')  # 두 집단의 분산이 같은지 검정
from scipy.stats import levene
levene_stat, leven_p = levene(male, female)
print(f"통계량:{levene_stat:.4f}, p-value:{leven_p:.4f}")
if leven_p > 0.05:
    print("분산이 같다고 할 수 있다.")
else:
    print("분산이 같다고 할 수 없다. 등분산 가정이 부적절")

# 만약 등분산성 가정이 부적절한 경우 Welch's t-test 사용을 권장

welch_result = stats.ttest_ind(male, female, equal_var=False)
print(welch_result)
# TtestResult(statistic=1.2331931, pvalue=0.2595335, df=6.61303)


