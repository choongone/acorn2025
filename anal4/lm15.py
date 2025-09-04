# 비선형회귀분석

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
x = np.array([257, 270, 294, 320, 342, 368, 396, 446, 480, 580])[:, np.newaxis]
print(x.shape)  # (10, 1)
y = np.array([236, 234, 253, 298, 314, 342, 360, 368, 390, 388])
# plt.scatter(x, y)
# plt.show()

# 일반회귀모델과 다항회귀모델 작성 후 비교
lr = LinearRegression()
pr = LinearRegression()

polyf = PolynomialFeatures(degree=2)
x_quad = polyf.fit_transform(x)

# 일반회귀모델 훈련
lr.fit(x, y)
x_fit = np.arange(250, 600, 10)[:, np.newaxis]
y_lin_fit = lr.predict(x_fit)
print(y_lin_fit)

# 다항(다변량)회귀모델 훈련
pr.fit(x_quad, y)
y_quad_fit = pr.predict(polyf.fit_transform(x_fit))
print(y_quad_fit)

# 시각화
plt.scatter(x, y, label='training point')
plt.plot(x_fit, y_lin_fit, label = 'linear fit', linestyle='--', c='red')   # 일반회귀
plt.plot(x_fit, y_quad_fit, label = 'quadratic fit', linestyle='-.', c='blue')  # 다항회귀
plt.legend()
plt.show()

y_lin_pred = lr.predict(x)
y_quad_pred = pr.predict(x_quad)

# 성능 비교 점수
print('MSE: 선형:%.3f, 다항:%.3f'%(mean_squared_error(y, y_lin_pred), \
                                mean_squared_error(y, y_quad_pred)))

print('설명력: 선형:%.3f, 다항:%.3f'%(r2_score(y, y_lin_pred), \
                                r2_score(y, y_quad_pred)))




