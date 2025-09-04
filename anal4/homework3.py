# 회귀분석 문제 2) 
# testdata에 저장된 student.csv 파일을 이용하여 세 과목 점수에 대한 회귀분석 모델을 만든다. 
# 이 회귀문제 모델을 이용하여 아래의 문제를 해결하시오.  수학점수를 종속변수로 하자.
#   - 국어 점수를 입력하면 수학 점수 예측
#   - 국어, 영어 점수를 입력하면 수학 점수 예측     https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/student.csv
import pandas as pd 
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/student.csv')
# print(data.head(3))
print(data.info())

# 종속 변수(y)와 독립 변수(X) 설정
y = data['수학']

# 2. 국어 점수만으로 수학 점수 예측 (단순 선형 회귀)
X1 = data[['국어']]
model1 = LinearRegression()
model1.fit(X1, y)

# 예측할 국어 점수 입력 (예시: 80점)
korean_score = int(input("국어점수를 입력하세요. :"))
predicted_math_score1 = model1.predict([[korean_score]])
print(f'국어점수 {korean_score}점일 때 예측된 수학 점수는{predicted_math_score1[0]:.2f}점 입니다.')

print('-' * 40)

# 3. 국어, 영어 점수로 수학 점수 예측 (다중 선형 회귀)
X2 = data[['국어', '영어']]
model2 = LinearRegression()
model2.fit(X2, y)

# 예측할 국어, 영어 점수 입력 (예시: 국어 80점, 영어 90점)
korean_score2 = int(input("국어점수를 입력하세요. :"))
english_score2 = int(input("영어점수를 입력하세요. :"))
predicted_math_score2 = model2.predict([[korean_score2, english_score2]])
print(f'국어 {korean_score2}점, 영어 {english_score2}점일 때 예측된 수학 점수는{predicted_math_score2[0]:.2f}점 입니다.')

"""
import statsmodels.formula.api as smf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api

# 문제1 : 국어 점수를 입력하면 수학 점수 예측
df = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/student.csv')
df = df.drop(columns=['이름'])
print(df)   # 상관계수 행렬은 문자는 안된다
print(df.info())
print(df.corr())
#           국어       영어      수학
# 국어  1.000000  0.915188  0.766263
# 영어  0.915188  1.000000  0.809668
# 수학  0.766263  0.809668  1.000000
print(np.corrcoef(df.국어,df.수학)) # 피어슨 상관계수 0.76626266 상관관계가 강한 편

result1=smf.ols(formula='수학~국어',data=df).fit()
print(result1.summary())
#                             OLS Regression Results
# ==============================================================================
# Dep. Variable:                     수학  R-squared:                       0.587(R²값)
# Model:                            OLS   Adj. R-squared:                  0.564
# Method:                 Least Squares   F-statistic:                     25.60
# Date:                Mon, 25 Aug 2025   Prob (F-statistic):           8.16e-05
# Time:                        12:54:50   Log-Likelihood:                -76.543
# No. Observations:                  20   AIC:                             157.1
# Df Residuals:                      18   BIC:                             159.1
# Df Model:                           1
# Covariance Type:            nonrobust
# ==============================================================================
#                  coef    std err          t      P>|t|      [0.025      0.975]
# ------------------------------------------------------------------------------
# Intercept     32.1069(y절편)     8.628      3.721      0.002      13.981      50.233
# 국어           0.5705(기울기)      0.113      5.060      0.000       0.334       0.807
# ==============================================================================
# Omnibus:                        1.833   Durbin-Watson:                   2.366
# Prob(Omnibus):                  0.400   Jarque-Bera (JB):                0.718
# Skew:                          -0.438   Prob(JB):                        0.698
# Kurtosis:                       3.310   Cond. No.                         252.
# ==============================================================================
print('결정계수',result1.rsquared)    # 결정계수 : 0.5871584576511684
print('pvalue',result1.pvalues[1])   #  pvalue(8.160795225697216e-05) < 0.05 유의한 모델
print('국어 55점에 대한 수학 예측 : ',0.5705*55+32.1069)   # 63.4844
print('국어 90점에 대한 수학 예측 : ',0.5705*90+32.1069)   # 83.4519
y_pred1=result1.predict(pd.DataFrame({'국어':[55]}))
y_pred2=result1.predict(pd.DataFrame({'국어':[90]}))
print('국어 55점에 대한 수학 예측 : ',y_pred1)   # 63.487159
print('국어 90점에 대한 수학 예측 : ',y_pred2)   # 83.456401


# 문제2 : 국어, 영어 점수를 입력하면 수학 점수 예측
result2=smf.ols(formula='수학~국어+영어',data=df).fit()
print(result2.summary())
#   OLS Regression Results
# ==============================================================================
# Dep. Variable:                     수학   R-squared:                       0.659(R²값)
# Model:                            OLS   Adj. R-squared:                  0.619
# Method:                 Least Squares   F-statistic:                     16.46
# Date:                Mon, 25 Aug 2025   Prob (F-statistic):           0.000105
# Time:                        12:58:07   Log-Likelihood:                -74.617
# No. Observations:                  20   AIC:                             155.2
# Df Residuals:                      17   BIC:                             158.2
# Df Model:                           2
# Covariance Type:            nonrobust
# ==============================================================================
#                  coef    std err          t      P>|t|      [0.025      0.975]
# ------------------------------------------------------------------------------
# Intercept     22.6238      9.482      2.386      0.029       2.618      42.629
# 국어             0.1158      0.261      0.443      0.663      -0.436       0.667
# 영어             0.5942      0.313      1.900      0.074      -0.066       1.254
# ==============================================================================
# Omnibus:                        6.313   Durbin-Watson:                   2.163
# Prob(Omnibus):                  0.043   Jarque-Bera (JB):                3.824
# Skew:                          -0.927   Prob(JB):                        0.148
# Kurtosis:                       4.073   Cond. No.                         412.
# ==============================================================================
print('결정계수',result2.rsquared)        # 결정계수 : 0.6586624558145443 상관관계가 강한 편
print('pvalue 국어',result2.pvalues[1])   # 국어 0.663 > 0.05 유의하지않은 모델
print('pvalue 영어',result2.pvalues[2])   # 영어 0.074 > 0.05 유의하지않은 모델
print('국어 55점, 영어 65점에 대한 수학 예측 : ',0.1158*55+0.5942*65+22.6238)   # 67.6158
print('국어 90점, 영어 85점에 대한 수학 예측 : ',0.1158*90+0.5942*85+22.6238)   # 83.5528
y_pred3=result2.predict(pd.DataFrame({'국어':[55],'영어':[65]}))
y_pred4=result2.predict(pd.DataFrame({'국어':[90],'영어':[85]}))
print('국어 55점, 영어 65점에 대한 수학 예측 : ',y_pred3)   # 67.61614
print('국어 90점, 영어 85점에 대한 수학 예측 : ',y_pred4)   # 83.55347
"""