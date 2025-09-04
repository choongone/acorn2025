# 차원축소(PCA - 주성분 분석)
# 독립성이 의심되는 파일들은 차원축소 할 수 있음 

import numpy as np 
import pandas as pd
# 독립변수(feature)
x1 = [95, 91, 66, 94, 68]
x2 = [56, 27, 25, 1, 9]
x3 = [57, 34, 9, 79, 4]
x = np.stack((x1, x2, x3), axis=0)
print(x)

x = pd.DataFrame(x.T, columns=['x1', 'x2', 'x3'])
print(x)

print('표준화 처리')
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_std = scaler.fit_transform(x)
print(x_std)    # 표준화한 값
print(scaler.inverse_transform(x_std))  # 표준화를 원복(다시 원상복구)

print('PCA 처리')
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
print(pca.fit_transform(x_std))
print(pca.inverse_transform(pca.fit_transform(x_std)))
print(scaler.inverse_transform(pca.inverse_transform(pca.fit_transform(x_std))))    # 원상복구  
# 주성분 분석 -> 다시 복구하더라도 일부 데이터가 손상됨(원래의 값과 매우 유사한 값으로 돌아가는 것)     [95. 56. 57.] -> [96.32707571 55.48056608 54.24044153]

print('와인 데이터로 분류(RandomForest) 연습 - PCA 전과 후로 나누어 실습')
from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics
from sklearn.model_selection import train_test_split
import pandas as pd
datas = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/wine.csv', header=None)
print(datas[:2])
# 1-fixed acidity 2-volatile acidity 3-citric acid 4-residual sugar 5-chlorides 6-free sulfur dioxide 7-total sulfur dioxide 8-density 9-pH 10-sulphates
# 11-alcoholOutput variable (based on sensory data):12-quality (score between 0 and 10)

x=np.array(datas.iloc[:, 0:12])
y=np.array(datas.iloc[:, 12])
print(x[:2])
print(y[:2], set(y))    # 1:red, 0:white

train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3, random_state=12)
print(train_x.shape, test_x.shape, train_y.shape, test_y.shape)     # (4547, 12) (1950, 12) (4547,) (1950,)
model = RandomForestClassifier(criterion='entropy', n_estimators=100).fit(train_x, train_y)
pred = model.predict(test_x)
print('pred : ', pred[:5])      # [0 0 0 1 0]
print('acc : ', sklearn.metrics.accuracy_score(test_y, pred))   #  0.9964102564102564
print('**' * 10)

pca = PCA(n_components=3)
x_pca = pca.fit_transform(x)
print(x[:2])
print(x_pca[:2])
train_x, test_x, train_y, test_y = train_test_split(x_pca, y, test_size=0.3, random_state=12)  # x대신 x_pca
model2 = RandomForestClassifier(criterion='entropy', n_estimators=100).fit(train_x, train_y)
pred2 = model2.predict(test_x)
print('pred2 : ', pred2[:5])      # [0 0 0 0 1] 
print('acc2 : ', sklearn.metrics.accuracy_score(test_y, pred2))    # 0.9579487179487179












