# RandomForest 분류/예측 알고리즘 
# 분류 알고리즘으로 titanic dataset 사용해  이진분류 
# Bagging 사용 : 데이터 샘플링(bootstrap)을 통해 모델을 학습시키고, 결과를 집계(Aggregating)
# 참고 : 우수한 성능을 원한다면 Boosting, 오버피팅이 걱정된다면 Bagging 추천 

# titanic dataset : feature (pclass, age, sex), label (survived)
import pandas as pd
import numpy as np 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score

df = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/titanic_data.csv')
print(df.head(2))
print(df.info())
print(df.isnull().any())
# pclass, age, sex 중 age에 null이 존재 
df = df.dropna(subset=['Pclass', 'Age', 'Sex'])     # pclass, age, sex에서 null이 있으면 잘라버리기 
print(df.shape)     # (714, 12)

# feature, label로 분리 
df_x = df[['Pclass', 'Age', 'Sex']].copy()
print(df_x.head(2))     # Sex를 숫자화
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
df_x.loc[:, 'Sex'] = encoder.fit_transform(df_x['Sex'])
print(df_x.head(2))

df_y = df['Survived']   # 종속변수 
print(df_y.head(2))
print()
train_x, test_x, train_y, test_y = train_test_split(df_x, df_y, test_size=0.3, random_state=12)

model = RandomForestClassifier(criterion='entropy', n_estimators=500)   # 의사결정 트리 갯수 : 500
model.fit(train_x, train_y)
pred = model.predict(test_x)
print('예측값 : ', pred[:10])               # 예측값 :  [1 0 0 0 0 0 1 1 0 1]
print('실제값 : ', np.array(test_y[:10]))   # 실제값 :  [1 0 0 0 1 0 1 1 0 1]
print('맞춘 갯수 : ', sum(test_y == pred))
print('전체 대비 맞춘 비율 : ', sum(test_y == pred) / len(test_y))
print('분류 정확도 : ', accuracy_score(test_y, pred))

# k-fold
cross_vali = cross_val_score(model, df_x, df_y, cv = 5)
print(cross_vali)
print('교차 검증 평균 정확도 : ', np.round(np.mean(cross_vali), 5))
print()
# 중요변수 확인 
print('특성(변수) 중요도 : ', model.feature_importances_)

import matplotlib.pyplot as plt
n_features = df_x.shape[1]
plt.barh(range(n_features), model.feature_importances_, align='center')
plt.xlabel('feature_importancesscore')
plt.ylabel('features')
plt.yticks(np.arange(n_features), df_x.columns)
plt.ylim(-1, n_features)
plt.show()
plt.close()

















