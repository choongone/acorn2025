# 의사결정나무(Decision Tree) - CART
# 분류(Classification)와 회귀(Regression) 모두 가능
# 비모수적 모델 → 선형성, 정규성, 등분산성 등의 가정 필요 없음
# 단점 : 과적합 발생 가능, 과적합 시 예측 정확도 낮아질 수 있음

import matplotlib.pyplot as plt
from sklearn import tree
import numpy as np
import pandas as pd

# 1. 데이터 준비
# 키(height)와 머리카락 길이(hair_length)로 성별(man/woman) 구분
x = [[180, 15], [177, 42], [156, 35], [174, 65], [161, 28],
     [160, 5], [170, 12], [176, 75], [170, 22], [175, 28]]
y = ['man', 'woman', 'woman', 'man', 'woman', 
     'woman', 'man', 'man', 'man', 'woman']

feature_names = ['height', 'hair_length']  # 특성 이름
class_names = ['man', 'woman']            # 클래스 이름

# 2. 모델 생성
model = tree.DecisionTreeClassifier(
    criterion='entropy',   # 정보 이득 기준(entropy), gini도 가능
    max_depth=3,           # 트리 최대 깊이 제한 (과적합 방지)
    random_state=0
)
model.fit(x, y)            # 학습 데이터로 모델 학습

# 3. 모델 옵션 참고
# max_depth : 트리의 최대 깊이
# min_samples_split : 노드를 분할하기 위한 최소 샘플 수
# min_samples_leaf : 리프 노드가 가져야 하는 최소 샘플 수
# max_features : 각 분할에서 고려할 특성 수

# 4. 학습 데이터 정확도 확인
print('훈련 데이터 정확도 : {:.3f}'.format(model.score(x, y)))
print('예측 결과 : ', model.predict(x))   # 학습 데이터 예측
print('실 제 값 : ', y)                   # 실제 값

# 5. 새로운 자료로 분류 예측
new_data = [[199, 60]]
print('새 데이터 예측 결과 : ', model.predict(new_data))

# 6. 결정트리 시각화
plt.figure(figsize=(10, 6))
tree.plot_tree(
    model, 
    feature_names=feature_names, 
    class_names=class_names,
    filled=True,      # 노드 색상 채우기
    rounded=True,     # 모서리 둥글게
    fontsize=12
)
plt.show()
