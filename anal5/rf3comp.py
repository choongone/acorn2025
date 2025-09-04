# titanic dataset : LogisticRegression, DecisionTreeClassifier, RandomForestClassifier 비교
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/titanic_data.csv')
print(df.head(2))
df.drop(columns=['PassengerId', 'Name', 'Ticket'], inplace=True)
print(df.head(2), df.shape)
print(df.describe())
# print(df.isnull().sum())

# Nill 처리 : 평균 또는 'N' 으로 변경 
df['Age'].fillna(df['Age'].mean(), inplace=True)
df['Cabin'].fillna('N', inplace=True)
df['Embarked'].fillna('N', inplace=True)
# print(df. isnull().sum())
print(df.info())

print('Sex : ', df['Sex'].value_counts())
print('Cabin : ', df['Cabin'].value_counts())
print('Embarked : ', df['Embarked'].value_counts())
df['Cabin'] = df['Cabin'].str[:1]
print(df.head(3))
print()
# 성별이 생존 확률에 어던 영향을 미쳤는지 확인 
print(df.groupby(['Sex', 'Survived'])['Survived'].count())
print('여성 생존율 : ', 233 / (81 + 233)) 
print('남성 생존쥴 : ', 109 / (468 + 109))
 
sns.barplot(x='Sex', y = 'Survived', data = df, errorbar=('ci', 95))
# plt.show()

# 성별 기준으로 Pclass별 생존 확률
sns.barplot(x='Pclass', y = 'Survived', hue = 'Sex', data = df)  # 성별기준 
# plt.show()

# 나이별 기준으로 생존 확률
def getAgeFunc(age):
    msg = ''
    if age <= -1: msg = 'unknown' 
    elif age <= 5: msg = 'baby'
    elif age <= 18: msg = 'teenager'
    elif age <= 65: msg = 'adult'
    else: msg = 'elder'
    return msg

df['Age_category'] = df['Age'].apply(lambda a:getAgeFunc(a))
print(df.head(2))

sns.barplot(x='Age_category', y = 'Survived', hue = 'Sex', data = df, order=['unknown', 'baby', 'teenager', 'adult', 'elder'])  
# plt.show()
del df['Age_category']

# 문자열 자료를 숫자화 
from sklearn import preprocessing
def labelIncoder(datas):
    cols = ['Cabin', 'Sex', 'Embarked']
    for c in cols:
        lab = preprocessing.LabelEncoder()
        lab = lab.fit(datas[c])
        datas[c] = lab.transform(datas[c])
    return datas 

df = labelIncoder(df)
print(df.head(3))
print(df['Cabin'].unique())         # [7 2 4 6 3 0 1 5 8]
print(df['Sex'].unique())           # [1 0]
print(df['Embarked'].unique())      # [3 0 2 1]

print()
# feature / label
feature_df = df.drop(['Survived'], axis='columns')
label_df = df['Survived']
print(feature_df.head(2))
print(label_df.head(2))

x_train, x_test, y_train, y_test = train_test_split(feature_df, label_df, test_size=0.2, random_state=1)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

logmodel = LogisticRegression(solver='lbfgs', max_iter=500).fit(x_train, y_train)
demodel = DecisionTreeClassifier().fit(x_train, y_train)
rfmodel = RandomForestClassifier().fit(x_train, y_train)

logpred = logmodel.predict(x_test)
print('logmodel acc:{0:.5f}'.format(accuracy_score(y_test, logpred)))
depred = demodel.predict(x_test)
print('demodel acc:{0:.5f}'.format(accuracy_score(y_test, depred)))
rfpred = rfmodel.predict(x_test)
print('rfmodel acc:{0:.5f}'.format(accuracy_score(y_test, rfpred)))