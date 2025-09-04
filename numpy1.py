# 기본 통계 함수를 직접 작성 : 평균, 분산, 표준편차 
# 평균으로부터 데이터들이 얼마나 떨어져 있는지 나타내는 것 
# 데이터들이 평균의 값과 비슷해질 수 있도록 노력할 것 

grades = [1, 3, -2, 4]

def grades_sum(grades):
    tot = 0
    for g in grades:
        tot += g
    return tot

print(grades_sum(grades))

def grades_ave(grades):
    ave = grades_sum(grades) / len(grades)
    return ave 

print(grades_ave(grades))

def grades_variance(grades):
    ave = grades_ave(grades)    # 위에 있는 ave와 다른 ave임
    vari = 0
    for su in grades:
        vari += (su - ave)**2
    return vari / len(grades)

print(grades_variance(grades))

def grades_std(grades):
    return grades_variance(grades) ** 0.5

print(grades_std(grades))

print('**' * 10)
import numpy as np
print('합은 ', np.sum(grades))
print('평균은 ', np.mean(grades)) # 산술적인 평균은 mean을 주로 사용 # 웬만해서는 평균으로 사용 
# print('평균은 ', np.average(grades)) # 가중평균을 구할 수 있음 
print('분산은 ', np.var(grades))
print('표준편차 ', np.std(grades))
