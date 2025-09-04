#  이원카이제곱
# 동질성:
# 검정 두 집단의 분포가 동일한가 다른 분포인가를 검증하는 방법. 두 집단 이상에서 각 
# 동일한가를 검정하게 됨. 두개 이상의 버주형 자료가 동일한 분포를 갖는 모집단에서 추출

# 검정실습 1. 교육방법에 따른 교육생들의 만족도 분석 동질성 검정 survey_method csv
import pandas as pd
import scipy.stats as stats

data = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/survey_method.csv')
print(data.head(3))
print(data['method'].unique)    # [1 2 3]
print(set(data['survey']))      # {1, 2, 3, 4, 5}

ctab = pd.crosstab(index=data['method'], columns=data['survey'])
ctab.columns = ['매우만족', '만족', '보통', '불만족', '매우불만족']
ctab.index = ['방법1', '방법2', '방법3']
print(ctab)

chi2, p, ddof, _ = stats.chi2_contingency(ctab)
msg = 'test statistic:{}, p-value:{}, df:{}'
print(msg.format(chi2, p, ddof))    
# test statistic:6.544667820529891, p-value:0.5864574374550608, df:8
# 해석: 유의수준 0.05 < p-value: 0.5864574374550608 이므로 귀무가설 채택.

print('---------------')
# 동질성
# 검정실습2) 연령대별sns이용률의동질성검정
# 20대에서 40대까지 연령대별로 서로 조금씩 그 특성이 다른 SNS서비스들에 대해 이용현황을 조사한 자료를 바탕으로 연령대별로 홍보 전략을 세우고자 한다.
# 연령대별로 이용현황이 서로 동일한지 검정해보도록 하자.
# 귀무 가설 : 연령대별로 sns 서비스별 이용 현황은 동일함. 
# 귀무 가설 : 연령대별로 sns 서비스별 이용 현황은 동일하지 않음.


data2 = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/snsbyage.csv')
print(data2.head(3))
print(data2['age'].unique())    # [1 2 3]
print(data2['service'].unique())    # ['F' 'T' 'K' 'C' 'E']

ctab2 = pd.crosstab(index=data2['age'], columns=data2['service']) # margins=True)  # margins -> 소계 확인
print(ctab2)
chi2, p, ddof, _ = stats.chi2_contingency(ctab2)
msg = 'test statistic:{}, p-value:{}, df:{}'
print(msg.format(chi2, p, ddof))    
# test statistic:102.75202494484225, p-value:1.1679064204212775e-18, df:8
# 해석 : 유의수준 0.05 > p-value:0.000 이므로 귀무가설 기각.

# 사실 위 데이터는 샘플데이터임. 
# 그런데 샘플링 연습을 위해 위 데이터를 모집단이라 가정하고 표본을 추출해 처리해보자.
sample_data = data2.sample(n=50, replace=True, random_state=1)  # random_state=1 -> 랜덤값을 고정 
print(len(sample_data))
ctab3 = pd.crosstab(index=sample_data['age'], columns=sample_data['service']) 
print(ctab2)
chi2, p, ddof, _ = stats.chi2_contingency(ctab3)
msg = 'test statistic:{}, p-value:{}, df:{}'
print(msg.format(chi2, p, ddof))








