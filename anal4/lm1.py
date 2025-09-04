# 최소 제곱해를 선형 행렬 방정식으로 구하기

import numpy as np
import matplotlib.pyplot as plt
plt.rc('font', family = 'Malgun Gothic')

x = np.array([0, 1, 2, 3])
y = np.array([-1, 0.2, 0.5, 2.1])
# plt.scatter(x, y)
# plt.show()

A = np.vstack([x, np.ones(len(x))]).T
print(A)        #  [[0. 1.]
                #   [1. 1.]
                #   [2. 1.]
                #   [3. 1.]]

import numpy.linalg as lin
# y = wx + b라는 1차방정식의 w, b?
w, b = lin.lstsq(A, y, rcond=None)[0]    # 최소제곱법 연산   # rcond=None -> 경고 안떨어짐 
# 최소제곱법 : 잔차 제곱의 총합이 최소가 되는 값을 얻을 수 있음 
print('w(weight, 기울기, slope) :', w)
print('b(bias, 절편, 편향, intercept) :', b)
# y = 0.9599999x + (-0.989999)      # 단순선형회귀수식(모델)

plt.scatter(x, y)
plt.plot(x, w * x + b, label='실제값')
plt.legend()
plt.show()
# 수식으로 예측값 얻기 
print( w * 1 + b)   # -0.029999(예측값) - 0.2(실제값) = 잔차, 오차, 손실, 에러 
# 예측값 -> 확률일 뿐 그대로 믿으면 X




