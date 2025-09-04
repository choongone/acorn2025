# 공분산, 상관계수
# 공분산 -> 두 변수의 패턴을 확인하기 위해 공분산을 사용. 단위 크기에 영향을 받음. 
# 공분산을 표준화 -> 상관계수    (-1 ~ 0 ~ 1)  +-1에 근사하면 관계가 강함
import numpy as np
# 공분산 : 패턴의 방향은 알겠으나 구체적인 크기를 표현하는 것은 곤란  
print(np.cov(np.arange(1, 6), np.arange(2, 7)))                         # 같은 방향으로 늘어남 
print(np.cov(np.arange(10, 60, 10), np.arange(20, 70, 10)))             # 값 고정
print(np.cov(np.arange(100, 600, 100), np.arange(200, 700, 100)))       # 늘어났다가 감소 
print(np.cov(np.arange(1, 6), (3, 3, 3, 3, 3))) 
print(np.cov(np.arange(1, 6), np.arange(6, 1, -1))) 
print('------------------------')
x = [8,3,6,6,9,4,3,9,3,4]
print('x의 평균: ', np.mean(x))
print('x의 분산: ', np.var(x))  # 평균과의 거리와 관련이 있음
y = [6,2,4,6,9,5,1,8,4,5]
print('y의 평균: ', np.mean(y))
print('y의 분산: ', np.var(y))

import matplotlib.pyplot as plt
# plt.scatter(x, y)
# plt.show()
print('x, y 공분산 : ', np.cov(x, y))       # 5.2   # 52.2 (y값에 0하나씩 더 붙였을 때)
print('x, y 공분산 : ', np.cov(x, y)[0,1])
print()
print('x, y의 상관계수 : ', np.corrcoef(x, y))      # 0.8663 (y값들이 바뀌어도 상관계수는 동일)
print('x, y의 상관계수 : ', np.corrcoef(x, y)[0, 1])

# 참고 : 비선형인 경우는 일반적인 상관계수 방법으르 사용하면 X
m = [-3,-2, -1, 0, 1, 2, 3]
n = [9, 4, 1, 0, 1, 4, 9]
plt.scatter(m, n)
plt.show()
print('m, n 상관계수 : ', np.corrcoef(m, n)[0, 1])  # 무의미한 작업 









