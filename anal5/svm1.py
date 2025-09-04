# SVM : 비확률적 이진 선형 분류모델 작성 가능 
# 직선적 분류 뿐 아니라 커널트릭을 이용해 비선형 분류도 가능 
# 커널(kernels) : 선형분류가 어려운 저차원 자료를 고차원 공간으로 매핑해서 분류

# LogisticRegression과 SVM으로 XOR 연산 처리 결과 분류 가능 확인 

from sklearn.linear_model import LogisticRegression
from sklearn import  svm, metrics
import pandas as pd
import numpy as np

x_data = [
    [0, 0, 0], 
    [0, 1, 1], 
    [1, 0, 1], 
    [1, 1, 0]
]
x_df = pd.DataFrame(x_data)
feature = np.array(x_df.iloc[:, 0:2])
label = np.array(x_df.iloc[:, 2])
print(feature)
print(label)

# model = LogisticRegression()
model = svm.SVC()
model.fit(feature, label)
pred = model.predict(feature)
print('예측값 : ', pred)
print('실제값 : ', label)
print('정확도 : ', metrics.accuracy_score(label, pred))












