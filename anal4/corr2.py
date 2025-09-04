# 공분산 / 상관계수 확인 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
plt.rc('font', family='Malgun Gothic')

data = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/drinking_water.csv')
print(data.head(3))
print(data.describe())
print()
print(np.std(data.친밀도))
print(np.std(data.적절성))
print(np.std(data.만족도))

# plt.hist([np.std(data.친밀도), np.std(data.적절성), np.std(data.만족도)])
# plt.show()

print('------공분산------')
print(np.cov(data.친밀도, data.적절성)) # numpy는 두 개씩만 확인할 수 있음 
print(np.cov(data.친밀도, data.만족도))
print(data.cov())   # DataFrame으로 공분산 출력

print('------상관계수------')
print(np.corrcoef(data.친밀도, data.적절성))
print(np.corrcoef(data.친밀도, data.만족도))
print(data.corr())  # DataFrame으로 상관계수 출력 
print(data.corr(method='pearson'))      # 변수가 등간, 비율 척도일 때 
print(data.corr(method='spearman'))     # 변수가 서열 척도일 때 
print(data.corr(method='kendall'))      

# ex) 만족도에 대한 다른 특성(변수)
co_re = data.corr()
print(co_re['만족도'].sort_values(ascending=False))
# 만족도    1.000000
# 적절성    0.766853
# 친밀도    0.467145

# 시각화 
data.plot(kind='scatter', x='만족도', y='적절성')
plt.show()

from pandas.plotting import scatter_matrix
attr = ['친밀도', '적절성', '만족도']
scatter_matrix(data[attr], figsize=(10, 6))
plt.show()

import seaborn as sns
sns.heatmap(data.corr())
plt.show()




 