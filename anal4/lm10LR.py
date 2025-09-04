# sklearn 모듈의 LinearRegression 클래스 사용
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, explained_variance_score, mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt

sample_size = 100 
np.random.seed(1)

# 편차가 없는 데이터 생성 
x = np.random.normal(0, 10, sample_size)
y = np.random.normal(0, 10, sample_size) + x * 60
print(x[:5])
print(y[:5])
print('상관계수 : ', np.corrcoef(x, y))

scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(x.reshape(-1, 1))
print('x_scaled : ', x_scaled)

# plt.scatter(x_scaled, y)
# plt.show() 

# 독립변수를 가지고 종속변수를 예측

model = LinearRegression().fit(x_scaled, y)
print(model)
print('계수(slope): ', model.coef_)     # 회귀계수(각 독립변수가 종속변수에 미치는 영향에 관련된 값)
print('절편(intercept): ', model.intercept_)
print('결정계수(R^2): ', model.score(x_scaled, y))      # 설명력 : 훈련 데이터 기준 
# y = wx + b    <==  계수 * x  + 절편
y_pred = model.predict(x_scaled)
print('예측값(y^)', y_pred[:5])     # [ 977.62741972 -366.1674945  -315.93693523 -643.3349412   521.54054659]
print('실제값(y)', y[:5])           # [ 970.13593255 -354.80877114 -312.86813494 -637.84538806  508.29545914]
# model.summary() 지원X

print()
# 선형회귀는 MAE(평균 절대 오차), MSE(평균 제곱 오차, RMSE(평균 제곱근 오차), R²(결정계수))
# 모델 성능 파악용 함수 작성 
def RegScoreFunc(y_true, y_pred):
    print('R^2_score(결정계수):{}'.format(r2_score(y_true, y_pred)))
    print('설명분산점수:{}'.format(explained_variance_score(y_true, y_pred)))
    print('mean_squared_error(평균제곱오차):{}'.format(mean_squared_error(y_true, y_pred)))
RegScoreFunc(y, y_pred)
# R^2_score(결정계수):0.9996956382642653
# 설명분산점수:0.9996956382642653
# mean_squared_error(평균제곱오차):86.14795101998757

print('-------------')
# 2) 편차가 있는 데이터 생성 
x = np.random.normal(0, 1, sample_size)
y = np.random.normal(0, 500, sample_size) + x * 30
print(x[:5])
print(y[:5])
print('상관계수 : ', np.corrcoef(x, y))

scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(x.reshape(-1, 1))
print('x_scaled : ', x_scaled)
plt.scatter(x_scaled, y)
plt.show() 

model = LinearRegression().fit(x_scaled, y)
print(model)
y_pred = model.predict(x_scaled)
print('예측값(y^): ',x[:5]) # 예측값(y^):  [-0.40087819  0.82400562 -0.56230543  1.95487808 -1.33195167]
print('실제값(y): ',y[:5])  # 실제값(y):  [1020.86531436 -710.85829436 -431.95511059 -381.64245767 -179.50741077]

RegScoreFunc(y, y_pred)
# R^2_score(결정계수):1.6093526521765433e-05
# 설명분산점수:1.6093526521765433e-05
# mean_squared_error(평균제곱오차):282457.9703485092











