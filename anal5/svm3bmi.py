# BMI 식을 이용해 데이터 만들기 
# 비만도 계산은 몸무게를 키의 제곱으로 나눈 것 
# 예) 키:175, 몸무게:68  ===>   68 / ((170/100) * (170/100))
# print(68 / ((170/100) * (170/100))) # 23.529


import random
random.seed(12)

def calc_bmi(h, w):
    bmi = w / ((h / 100) ** 2)
    if bmi < 18.5: return 'thin'
    if bmi < 25.5: return 'normal'
    return 'fat'

# print(calc_bmi(170,70))
fp = open('bmi.csv', 'w')
fp.write('height,weight,label\n')

cnt = {'thin' : 0, 'normal' : 0, 'fat' : 0}

for i in range(50000):
    h = random.randint(150, 200)
    w = random.randint(35, 100)
    label = calc_bmi(h, w)
    cnt[label] += 1
    fp.write('{0},{1},{2}\n'.format(h, w, label))

fp.close()


# SVM으로 분류 모델
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd

tbl = pd.read_csv('bmi.csv')
print(tbl.head(3), tbl.shape)
print(tbl.info())

label = tbl['label']
print(label[:3])

w = tbl['weight'] / 100 # 정규화
print(w[:3].values)

h = tbl['height'] / 200
print(h[:3].values)

wh = pd.concat([w, h], axis = 1)
print(wh.head(3), wh.shape)

# label 정수화
label = label.map({'thin' : 0, 'normal' : 1, 'fat' : 2})
print(label[:3])

x_train, x_test, y_train, y_test = train_test_split(wh, label, test_size = 0.3, random_state = 1)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape) # (35000, 2) (15000, 2) (35000,) (15000,)

model = svm.SVC(C = 0.01, kernel = 'rbf', random_state = 1).fit(x_train, y_train)
pred = model.predict(x_test)
print('예측값 : ', pred[:10])             # [2 0 1 1 0 0 2 1 0 0]
print('실제값 : ', y_test[:10].values)    # [2 0 1 1 0 0 2 1 0 0]

ac_score = metrics.accuracy_score(y_test, pred)
print('accuracy : ', ac_score)

tbl2 = pd.read_csv('bmi.csv', index_col=2)
def scatterFunc(lbl, color):
    b = tbl2.loc[lbl]
    plt.scatter(b['weight'], b['height'], c=color, label=lbl)

scatterFunc('fat', 'red')
scatterFunc('normal', 'yellow')
scatterFunc('thin', 'blue')
plt.legend()
plt.show()

# 새로운 값으로 예측 
newData = pd.DataFrame({'weight':[69, 89],'height':[170, 170]})
newData['weight'] = newData['weight'] / 100
newData['height'] = newData['weight'] / 200
new_pred = model.predict(newData)
print('새로운 데이터에 대한 bmi는 ', new_pred)




