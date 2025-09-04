# [two-sample t 검정 : 문제1] 
# 다음 데이터는 동일한 상품의 포장지 색상에 따른 매출액에 대한 자료이다. 
# 포장지 색상에 따른 제품의 매출액에 차이가 존재하는지 검정하시오.
# blue : 70 68 82 78 72 68 67 68 88 60 80
# red : 60 65 55 58 67 59 61 68 77 66 66

# 귀무 : 포장지 색상에 따라 제품의 매출액에 차이가 존재하지 않는다.
# 대립 : 포장지 색상에 따라 제품의 매출액에 차이가 존재한다. 

from scipy.stats import levene
from scipy import stats
import pandas as pd
import numpy as np



blue = [70, 68, 82, 78, 72, 68, 67, 68, 88, 60, 80]
red = [60, 65, 55, 58, 67, 59, 61, 68, 77, 66, 66]
print(np.mean(blue), '  ', np.mean(red))        # 72.8181  |  63.8181

two_sample = stats.ttest_ind(blue, red, equal_var=True)
print(two_sample)       # TtestResult(statistic=2.92802, pvalue=0.00831, df=20.0)
# pvalue < 0.05 -> 귀무가설 채택X

print('\n====== 등분산 검정 ======')
levene_stat, levene_p = levene(blue, red)

print(f"통계량: {levene_stat:.5f}, p-value:{levene_p:.5f}")
if levene_p > 0.05: 
    print("분산이 같다고 할 수 있음")
else:
    print("분산이 같다고 할 수 없음")



# [two-sample t 검정 : 문제2]  
# 아래와 같은 자료 중에서 남자와 여자를 각각 15명씩 무작위로 비복원 추출하여 혈관 내의 콜레스테롤 양에 차이가 있는지를 검정하시오.
# 남자 : 0.9 2.2 1.6 2.8 4.2 3.7 2.6 2.9 3.3 1.2 3.2 2.7 3.8 4.5 4 2.2 0.8 0.5 0.3 5.3 5.7 2.3 9.8
# 여자 : 1.4 2.7 2.1 1.8 3.3 3.2 1.6 1.9 2.3 2.5 2.3 1.4 2.6 3.5 2.1 6.6 7.7 8.8 6.6 6.4

# 귀무 : 남녀 혈관 내의 콜레스테롤 양에 차이가 없다. 
# 대립: 남녀 혈관 내의 콜레스테롤 양에 차이가 있다. 


male = [0.9, 2.2, 1.6, 2.8, 4.2, 3.7, 2.6, 2.9, 3.3, 1.2, 3.2, 2.7, 3.8, 4.5, 4.0, 2.2, 0.8, 0.5, 0.3, 5.3, 5.7, 2.3, 9.8]
female = [1.4, 2.7, 2.1, 1.8, 3.3, 3.2, 1.6, 1.9, 2.3, 2.5, 2.3, 1.4, 2.6, 3.5, 2.1, 6.6, 7.7, 8.8, 6.6, 6.4]

male_sample = np.random.choice(male, 15, replace=False)
female_sample = np.random.choice(female, 15, replace=False)

print("무작위 남자 표본: ", male_sample)
print("무작위 여자 표본: ", female_sample)    

levene_pvalue = stats.levene(male_sample, female_sample).pvalue
print('등분산성 : , {leven_pvalue:.5f}')
      
if levene_pvalue < 0.05:
    equal_variance = False
    print("결론: 분산이 다르므로 Welch's T-test를 사용합니다.")
else:
    equal_variance = True
    print("결론: 분산이 같으므로 Student's T-test를 사용합니다.")


t_statistic, p_value = stats.ttest_ind(male_sample, female_sample, equal_var = equal_variance)

print("\n--- T-검정 결과 ---")
print(f"T-검정 통계량: {t_statistic:.4f}")
print(f"P-값: {p_value:.4f}")

if p_value < 0.05:
    print("\n결론: P-값이 유의수준 0.05보다 작으므로 귀무가설을 기각")
    print("따라서, 남녀 콜레스테롤 양의 평균에는 유의미한 차이가 있다고 볼 수 있다.")
else:
    print("\n최종 결론: P-값이 유의수준 0.05보다 크므로 귀무가설을 기각할 수 없음")
    print("따라서, 남녀 콜레스테롤 양의 평균에는 유의미한 차이가 없다고 볼 수 있다.")

"""
# 귀무: 남녀 두 집단 간 혈관 내의 콜레스테롤 양에 차이가 없다.
# 대립: 남녀 두 집단 간 혈관 내의 콜레스테롤 양에 차이가 있다.
남자 = [0.9, 2.2, 1.6, 2.8, 4.2, 3.7, 2.6, 2.9, 3.3, 1.2, 3.2, 2.7, 3.8, 4.5, 4, 2.2, 0.8, 0.5, 0.3, 5.3, 5.7, 2.3, 9.8]
여자 = [1.4, 2.7, 2.1, 1.8, 3.3, 3.2, 1.6, 1.9, 2.3, 2.5, 2.3, 1.4, 2.6, 3.5, 2.1, 6.6, 7.7, 8.8, 6.6, 6.4]
print(np.mean(남자), ' ', np.mean(여자))   # 3.06   3.54
print(len(남자))    # 23
print(len(여자))    # 20

샘플개수 = 15
남자샘플 = random.sample(남자, 샘플개수)    # 남자에서 15명씩 무작위로 비복원 추출
여자샘플 = random.sample(여자, 샘플개수)    # 여자에서 15명씩 무작위로 비복원 추출

two_sample = stats.ttest_ind(남자샘플, 여자샘플)
print(two_sample)
# TtestResult(statistic=-0.28675, pvalue=0.77641, df=28.0)
# 해석: pvalue=0.0.65162 > 0.05 이므로 귀무가설 채택
# 남녀 두 집단 간 혈관 내의 콜레스테롤 양에 차이가 없다.
"""

# [two-sample t 검정 : 문제3]
# DB에 저장된 jikwon 테이블에서 총무부, 영업부 직원의 연봉의 평균에 차이가 존재하는지 검정하시오.
# 연봉이 없는 직원은 해당 부서의 평균연봉으로 채워준다.

# 귀무: 총무부, 영업부 직원의 연봉의 평균에 차이가 없다.
# 대립: 총무부, 영업부 직원의 연봉의 평균에 차이가 있다.

import MySQLdb
import matplotlib.pyplot as plt

config = {
    'host':'127.0.0.1',
    'user':'root',
    'password':'1234',
    'database':'mydb',
    'port':3306,
    'charset':'utf8'
}

try:
    conn = MySQLdb.connect(**config)
    cursor = conn.cursor()
    # 총무부 연봉 추출
    sql = '''
        select jikwonpay
        from jikwon inner join buser
        on jikwon.busernum = buser.buserno
        where busername = '총무부';
    '''
    cursor.execute(sql)

    results = cursor.fetchall()
    총무부 = [result[0] for result in results]
    print('총무부 연봉: ', 총무부)

    # 영업부 연봉 추출
    sql = '''
        select jikwonpay
        from jikwon inner join buser
        on jikwon.busernum = buser.buserno
        where busername = '영업부';
    '''
    cursor.execute(sql)

    results = cursor.fetchall()
    영업부 = [result[0] for result in results]
    print('영업부 연봉: ', 영업부)

except Exception as e:
    print('처리 오류: ', e)
finally:
    conn.close()

plt.boxplot([총무부, 영업부], meanline=True, showmeans=True)
plt.show()
plt.close()

print(np.mean(총무부), ' ', np.mean(영업부))   # 5414.28   4908.33
two_sample = stats.ttest_ind(총무부, 영업부)
print(two_sample)
# TtestResult(statistic=0.45851, pvalue=0.65238, df=17.0)
# 해석: pvalue=0.65238 > 0.05 이므로 귀무가설 채택
# 귀무: 총무부, 영업부 직원의 연봉의 평균에 차이가 없다.












