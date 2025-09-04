# 1) 
# kaggle.com이 제공하는 'Red Wine quality' 분류 ( 0 - 10)

# Input variables (based on physicochemical tests):
#  1 - fixed acidity
#  2 - volatile acidity
#  3 - citric acid
#  4 - residual sugar
#  5 - chlorides
#  6 - free sulfur dioxide
#  7 - total sulfur dioxide
#  8 - density
#  9 - pH
#  10 - sulphates
#  11 - alcohol
#  Output variable (based on sensory data):
#  12 - quality (score between 0 and 10)
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle

df = pd.read_csv('winequality-red.csv')
print(df.head(2))
print(df.shape, df.info())
print(df.describe())

model = RandomForestClassifier(criterion='entropy', n_estimators=500, random_state=1)

x = df.drop('quality', axis=1)
y = df['quality']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(criterion='entropy', n_estimators=500, random_state=1)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

print('예측값 : ', y_pred)       
print('실제값 : ', np.array(y_test))

print('총 갯수:%d, 오류수:%d' % (len(y_test), (y_test != y_pred).sum()))

print('분류 정확도 : %.5f' % accuracy_score(y_test, y_pred))
print()

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

plt.figure(figsize=(10, 8))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

importances = model.feature_importances_
feature_names = x.columns
feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances}).sort_values(by='importance', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x='importance', y='feature', data=feature_importance_df)
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.grid(axis='x', linestyle='--')
plt.show()

pickle.dump(model, open('rf_model.sav', 'wb'))
del model 

read_model = pickle.load(open('rf_model.sav', 'rb'))

print('불러온 모델 정확도 : ', read_model.score(x_test, y_test))



# 2)
# 중환자 치료실에 입원 치료 받은 환자 200명의 생사 여부에 관련된 자료다.
# 종속변수 STA(환자 생사 여부)에 영향을 주는 주요 변수들을 이용해 검정 후에 해석하시오. 
# 모델 생성 후 입력자료 및 출력결과는 Django를 사용하시오.
# 예제 파일 : https://github.com/pykwon  ==>  patient.csv
# <변수설명>
#   STA : 환자 생사 여부 (0:생존, 1:사망)
#   AGE : 나이
#   SEX : 성별
#   RACE : 인종
#   SER : 중환자 치료실에서 받은 치료
#   CAN : 암 존재 여부
#   INF : 중환자 치료실에서의 감염 여부
#   CPR : 중환자 치료실 도착 전 CPR여부
#   HRA : 중환자 치료실에서의 심박수


# <변수설명>
#   STA : 환자 생사 여부 (0:생존, 1:사망)
#   AGE : 나이
#   SEX : 성별
#   RACE : 인종
#   SER : 중환자 치료실에서 받은 치료
#   CAN : 암 존재 여부
#   INF : 중환자 치료실에서의 감염 여부
#   CPR : 중환자 치료실 도착 전 CPR여부
#   HRA : 중환자 치료실에서의 심박수

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyparsing import col
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/patient.csv')
# print(df.head())
print(df.isnull().sum()) # 결측치 확인

feature_df = df.drop('STA', axis=1)
label_df = df['STA']
x_train, x_test, y_train, y_test = train_test_split(feature_df, label_df, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(x_train, y_train)

pred = model.predict(x_test)
accuracy = accuracy_score(y_test, pred)
print('예측값 : ', pred)
print('실제값 : ', np.array(y_test))
print('정확도 : ', accuracy)

# 중요 변수 알아보기
# 참고 : 중요 변수 알아보기
print('특성(변수) 중요도 :\n{}'.format(model.feature_importances_))
def plot_feature_importances(model, x):   # 특성 중요도 시각화
    n_features = x.shape[1]
    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), x.columns)
    plt.xlabel("attr importances")
    plt.ylabel("attr")
    plt.ylim(-1, n_features)
    plt.show()
    plt.close()

plot_feature_importances(model, x_train)

print('중요도 상위 3개 변수 :')
for i in model.feature_importances_.argsort()[-3:][::-1]:
    print(' - {} : {:.4f}'.format(x_train.columns[i], model.feature_importances_[i]))






