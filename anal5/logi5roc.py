# 분류모델 성능 평가 관련 
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd

x, y = make_classification(n_samples=100, n_features=2, n_redundant=0, random_state=123)  # 샘플수 100개, 독립변수 2개
print(x[:3])
print(y[:3])

import matplotlib.pyplot as plt
# plt.scatter(x[:, 0], x[:, 1])
# plt.show()

model = LogisticRegression().fit(x, y)
yhat = model.predict(x)
print('yhat : ', yhat[:3])

f_value = model.decision_function(x)
# 결정함수(판별함수, 불확실성 추정함수), 판별경계선 설정을 위한 샘플 자료 얻기
print('f_value : ', f_value[:10])

df = pd.DataFrame(np.vstack([f_value, yhat, y]).T, columns=["f", "yhat", "y"])  # .T -> 나오는 자료를 새로로 변경
print(df.head(3))

from sklearn.metrics import confusion_matrix
print(confusion_matrix(y, yhat))
acc = (44 + 44) / 100
recall = 44 / (44 + 4)
precision = 44 / (44 + 8)
specificity = 44 / (8 + 44)     # TN / (FP + TN) 
fallout = 8 / (8 + 44)          # 위양성율    FP / (FP + TN)
print('acc(정확도) : ', acc)                         # 0.88
print('recall(재현율) : ', recall)                   #  0.91666
print('precision(정밀도) : ', precision)             # 0.846153
print('spcificity(특이도) : ', specificity)          # 0.846153
print('fallout(위양성율) : ', fallout)               # 0.153846
print('fallout(위양성율) : ', 1 - specificity)       # 0.153846
# 정리하면 TPR은 1에 근사하면 좋고, FPR은 0에 근사하면 좋음 
print()
from sklearn import metrics
ac_sco = metrics.accuracy_score(y, yhat)
print('ac_sco : ', ac_sco)
cl_rep = metrics.classification_report(y, yhat)
print(cl_rep )
print()

fpr, tpr, thresholds = metrics.roc_curve(y, model.decision_function(x))
print('fpr : ', fpr)
print('tpr : ', tpr)
print('분류임계결정값 : ', thresholds)

# ROC 커브 시각화 
plt.plot(fpr, tpr, 'o-', label='LogisticRegression')
plt.plot([0, 1], [0, 1], 'k--', label='random classifier line(AUC 0.5)')
plt.plot([fallout], [recall], 'ro', ms=10)      # 위양성률과 재현율 값 출력
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.title('ROC curve')
plt.legend()
plt.show()

# AUC(Area Under the Curve) - ROC 커브의 면적 
# 1에 가까울수록 좋은 분류모델로 평가됨 
print('AUC : ', metrics.auc(fpr, tpr))










