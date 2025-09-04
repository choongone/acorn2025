# Logistic Regression
# 독립(feature, x):연속형, 종속(label, y):범주형
# 이항 분류(다항도 가능)
# 출력된 연속형(확률)자료를 logit 변환해, 최종적으로 sigmoid function에 의해 0 ~ 1 사이의 실수값이 나오는데 
# 0.5를 기준으로 0, 1로 분류한다.

# sigmoid function 살짝 맛보기
import math
def sigmoidFunc(x):
    return 1 / (1 + math.exp(-x))

# print(sigmoidFunc(3))
# print(sigmoidFunc(1))
# print(sigmoidFunc(-123))
# print(sigmoidFunc(0.123))

# mtcar dataset 사용
import statsmodels.api as sm
mtcardata = sm.datasets.get_rdataset('mtcars')
print(mtcardata.keys())
mtcars = mtcardata.data
print(mtcars.head(2))
# mpg, hp가 am(자동, 수동)에 영향을 준다.
mtcar = mtcars.loc[:, ['mpg', 'hp', 'am']]  # mpg, hp, am의 데이터만 가져온다.
print(mtcar.head(2))
print(mtcar['am'].unique())     # [1 0] 1은 수동 0은 자동

# 연비와 마력수에 따른 변속기 분류 모델 작성(수동, 자동)
# 모델 작성 방법1: logit()
import statsmodels.formula.api as smf
formula = 'am ~ hp + mpg'
model1 = smf.logit(formula=formula, data=mtcar).fit()
print(model1.summary()) # Logit Regression 결과표

# 예측값 / 실제값 출력
import numpy as np
import pandas as pd
# print('예측값: ', model1.predict())
pred = model1.predict(mtcar[:10])
print('예측값: ', pred.values)
print('예측값: ', np.around(pred.values))   # np.around가 if문 없이 0과 1로 나눠준다.
print('실제값: ', mtcar['am'][:10].values)
print('-'*100)
# 분류 모델의 정확도(accuracy) 확인
conf_tab = model1.pred_table()      # 수치에 대한 집계표
print('confusion matrix: \n',conf_tab)  
# [[16.  3.]    모델이 맞췄다는게 16개  모델이 틀렸다고 했는데 맞은건 3개
# [ 3. 10.]]    모델이 틀린걸 틀렸다고 예측한것이10개
print('분류 정확도: ', (16 + 10) / len(mtcar))      # 모델이맞춘갯수 / 전체갯수
print('분류 정확도: ', (conf_tab[0][0] + conf_tab[1][1]) / len(mtcar))  # 분류 정확도:  0.8125는 81프로의 확률로 맞춘다는 의미이다.

from sklearn.metrics import accuracy_score
pred2 = model1.predict(mtcar)
print("분류 정확도: ", accuracy_score(mtcar['am'], np.around(pred2)))

print('--------------------')
# 모델 작성 방법2 : glm()  -> 일반화된 선형모델 
model2 = smf.glm(formula=formula, data=mtcar, family=sm.families.Binomial()).fit()
print(model2)
print(model2.summary())     # hp, mpg는 0.05보다 큰가 안큰가를 확인
glm_pred = model2.predict(mtcar[:10])
print('glm 예측값 : ', np.around(glm_pred.values))  # [0. 0. 1. 0. 0. 0. 0. 1. 1. 0.]
print('glm 실제값 : ', mtcar['am'][:10].values)     # [1 1 1 0 0 0 0 0 0 0]

glm_pred2 = model2.predict(mtcar)
print("glm 분류 정확도: ", accuracy_score(mtcar['am'], np.around(glm_pred2)))

print('\n새로운 값(hp, mpg)으로 변속기(am) 분류 예측')
newdf = mtcar.iloc[:2].copy()
# print(newdf)
newdf['mpg'] = [10, 30]
newdf['hp'] = [120, 90]
print(newdf)
new_pred = model2.predict(newdf)
print('새로운 값(hp, mpg)에 대한 변속기는 ', np.around(new_pred.values))
print()
import pandas as pd
newdf2 = pd.DataFrame({'mpg':[10,30,50,5], 'hp':[80, 110, 130,50]})
new_pred2 = model2.predict(newdf2)
print('new_pred2 : ', new_pred2.values)
print('new_pred2 : ', np.around(new_pred2.values))  # 위의 값을 간략하게 줄여줌 
print('new_pred2 : ', np.rint(new_pred2.values))    # 위의 값을 간략하게 줄여줌








'''
                            Logit Regression Results
==============================================================================
Dep. Variable:                     am   No. Observations:                   32
Model:                          Logit   Df Residuals:                       29
Method:                           MLE   Df Model:                            2
Date:                Wed, 27 Aug 2025   Pseudo R-squ.:                  0.5551
Time:                        17:09:59   Log-Likelihood:                -9.6163
converged:                       True   LL-Null:                       -21.615
Covariance Type:            nonrobust   LLR p-value:                 6.153e-06
==============================================================================
                coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept    -33.6052     15.077     -2.229      0.026     -63.156      -4.055
hp             0.0550      0.027      2.045      0.041       0.002       0.108
mpg            1.2596      0.567      2.220      0.026       0.147       2.372
==============================================================================

p > |z|에 hp와 mpg의 값이 0.05보다 작다.
'''

