# 자전거 공유 시스템(워싱턴 D.C) 관련 파일로 시각화 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.rc('font', family='malgun gothic')
plt.rcParams['axes.unicode_minus'] = False

plt.style.use('ggplot')
# 데이터 수집 후 가공(EDA) - train.csv
train = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/data/train.csv', parse_dates=['datetime'])
print(train.shape)  # 10886, 12
print(train.columns)  # 리스트로 나옴
print(train.info())  # 각 컬럼의 구조가 나옴 
print(train.head(3))
pd.set_option('display.max_columns', 500)
# print(train.describe())
print(train.temp.describe())
print(train.isnull().sum())
# null이 포함된 열 확인용 시각화 모듈  # conda install missingno
import missingno as msno
# msno.matrix(train, figsize=(12, 5))
# plt.show()
# msno.bar(train)
# plt.show()

# 연월일시 데이터로 자전거 대여량 시각화
train['year'] = train['datetime'].dt.year
train['month'] = train['datetime'].dt.month
train['day'] = train['datetime'].dt.day
train['hour'] = train['datetime'].dt.hour
train['minute'] = train['datetime'].dt.minute
train['second'] = train['datetime'].dt.second
print(train.columns)
print(train.head(1))
figure,(ax1,ax2,ax3,ax4) = plt.subplots(nrows=1, ncols=4)
figure.set_size_inches(15, 5)
sns.barplot(data=train, x='year', y='count', ax=ax1)
sns.barplot(data=train, x='month', y='count', ax=ax2)
sns.barplot(data=train, x='day', y='count', ax=ax3)
sns.barplot(data=train, x='hour', y='count', ax=ax4)
ax1.set(ylabel='건수', title='연도별 대여량')
ax2.set(ylabel='월', title='월별 대여량')
ax3.set(ylabel='일', title='일별 대여량')
ax4.set(ylabel='시', title='시간별 대여량')
plt.show()

# Boixplot으로 시각화 - 대여량 - 계절별, 시간별, 근무일 여부에 따른 대여량 
fig, axes = plt.subplots(nrows=2, ncols=2)
fig.set_size_inches(12, 10)
sns.boxplot(data=train, y='count', orient='v', ax=axes[0][0])  # orient='h
sns.boxplot(data=train, y='count', x='season', orient='v', ax=axes[0][1]) 
sns.boxplot(data=train, y='count', x='hour', orient='v', ax=axes[1][0]) 
sns.boxplot(data=train, y='count', x='workingday', orient='v', ax=axes[1][1])
axes[0][0].set(ylabel='건수', title='대여량') 
axes[0][1].set(xlabel='계절별', title='계절별 대여량')
axes[1][0].set(xlabel='계절별', title='시간별 대여량')
axes[1][1].set(xlabel='근무일', title='근무일 여부에 따른 대여량')
plt.show()

