# 방법4 : linregress     model 만들어짐 
from scipy import stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# IQ에 따른 시험 점수 값 예측
score_iq = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/score_iq.csv')
print(score_iq.head(3))
print(score_iq.info())

x = score_iq.iq         # score_iq에서 iq를 뺀 것 
y = score_iq.score      # score_iq에서 score를 뺀 것

# 상관계수 확인
print(np.corrcoef(x, y)[0, 1])
print(score_iq.corr())
# plt.scatter(x, y)
# plt.show()

model = stats.linregress(x, y)
print(model)    
# LinregressResult(slope=np.float64(0.6514309527270075), intercept=np.float64(-2.8564471221974657), rvalue(결정계수)=np.float64(0.8822203446134699), 
# pvalue=np.float64(2.8476895206683644e-50), stderr=np.float64(0.028577934409305443), intercept_stderr=np.float64(3.546211918048538))
# pvalue < 0.05 -> 이 모델은 두 변수 간의 인과관계 O, 의미 있는 데이터
print('기울기 : ', model.slope)                     # 0.6514309527270075
print('절편 : ', model.intercept)                   # -2.8564471221974657
print('R² - 결정계수(설명력) : ', model.rvalue)      # 0.8822203446134699 : 독립변수가 종속변수를 88%정도 설명하고 있음 
print('p-value : ', model.pvalue)                  # 2.8476895206683644e-50 < 0.05 이므로 현재 모델은 유의함 (독립변수와 종속변수는 인과관계가 있음)
print('표준오차 : ', model.stderr)                  # 0.028577934409305443
# ŷ = wx + b => 0.6514309527270075 * x + (0.6514309527270075)

plt.scatter(x, y)
plt.plot(x, model.slope * x + model.intercept)
plt.show()

# 점수 예측 
print('점수 예측 : ', model.slope * 80 + model.intercept)
print('점수 예측 : ', model.slope * 120 + model.intercept)
# predict X 
print('점수 예측 : ', np.polyval([model.slope, model.intercept], np.array(score_iq['iq'][:5])))

print()
newdf = pd.DataFrame({'iq' :[55, 66, 77, 88, 150]})
print('점수 예측 : ', np.polyval([model.slope, model.intercept], newdf))






