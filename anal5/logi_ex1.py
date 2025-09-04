import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_curve, auc
import matplotlib.pyplot as plt

# 데이터 불러오기
url = "https://raw.githubusercontent.com/pykwon/python/master/testdata_utf8/bodycheck.csv"
df = pd.read_csv(url)

# 특징과 타겟 선택
X = df[['게임', 'TV시청']]
y = df['안경유무']

# 학습용 / 테스트용 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 로지스틱 회귀 모델 생성 및 학습
model = LogisticRegression()
model.fit(X_train, y_train)

# 예측
y_pred = model.predict(X_test)

# 모델 평가
print(" 모델 정확도(accuracy):", accuracy_score(y_test, y_pred))
print("\n confusion_matrix:\n", confusion_matrix(y_test, y_pred))
print("\n classification_report:\n", classification_report(y_test, y_pred))

# ROC Curve 및 AUC
y_score = model.decision_function(X_test)
fpr, tpr, thresholds = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC={roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Random')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curve')
plt.legend()
plt.show()

print(" AUC 값:", roc_auc)

# 새로운 데이터 입력 받아 예측
print("\n=== 새로운 데이터 예측 ===")
game = int(input("게임 시간 입력(정수): "))
tv = int(input("TV 시청 시간 입력(정수): "))

new_data = [[game, tv]]
pred = model.predict(new_data)
prob = model.predict_proba(new_data)

print(f"입력 데이터 → 게임:{game}, TV시청:{tv}")
print("안경 착용 예측(0 = 착용X, 1 = 착용O):", pred[0])
print("클래스별 확률:", prob)
