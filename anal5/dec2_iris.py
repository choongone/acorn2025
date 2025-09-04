from sklearn import datasets
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os
import pickle

# 한글 폰트 설정 및 마이너스 기호 처리
plt.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False

# 1. 데이터 로드
iris = datasets.load_iris()                      # iris 데이터셋 로드
x = iris.data[:, [2,3]]                          # 꽃잎 길이와 너비만 선택 (2,3 컬럼)
y = iris.target                                  # 타겟 라벨

# 2. 학습용 / 테스트용 데이터 분리
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=0          # 30%는 테스트용, 랜덤 시드 0
)

# 3. DecisionTreeClassifier 생성
model = DecisionTreeClassifier(
    criterion='entropy',                          # 정보 이득(entropy) 기준
    max_depth=5,                                  # 트리 최대 깊이 제한
    min_samples_leaf=5,                           # 리프 노드 최소 샘플 수
    random_state=0
)
model.fit(x_train, y_train)                       # 학습 데이터로 학습

# 4. 예측 및 정확도 확인
y_pred = model.predict(x_test)
print('예측값 :', y_pred)
print('실제값 :', y_test)
print('총 갯수:%d, 오류수:%d' % (y_test.shape[0], (y_test != y_pred).sum()))
print('분류정확도 : %.5f' % accuracy_score(y_test, y_pred))

# 5. confusion matrix
con_mat = pd.crosstab(y_test, y_pred, rownames=['예측값'], colnames=['관측값'])
print(con_mat)                                  # 오차 행렬 확인

# 6. 모델 저장 (폴더 생성 포함)
if not os.path.exists('anal8'):                 # 폴더 없으면 생성
    os.makedirs('anal8')
pickle.dump(model, open('anal8/logimodel.sav', 'wb'))  # 모델 저장

# 7. 모델 불러오기
read_model = pickle.load(open('anal8/logimodel.sav', 'rb'))  # 저장된 모델 불러오기

# 8. 새로운 데이터 예측
new_data = np.array([[5.1, 1.1], [1.1, 1.1], [6.1, 7.1]])  # 예측용 샘플
new_pred = read_model.predict(new_data)                     # 클래스 예측
print('new_data 예측 결과 :', new_pred)
print('예측 확률 :\n', read_model.predict_proba(new_data))  # 각 클래스별 확률

# 9. 결정 경계 시각화 함수
def plot_decisionFunc(X, y, classifier, test_idx=None, resolution=0.02, title=''):
    markers = ('s','x','o','^','v')                                # 마커 스타일
    colors = ('r', 'b', 'lightgray', 'gray', 'cyan')              # 색상
    cmap = ListedColormap(colors[:len(np.unique(y))])             # 컬러맵 생성

    # 결정 경계 범위 지정
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                         np.arange(x2_min, x2_max, resolution))
    
    # 예측값 계산 후 reshape
    Z = classifier.predict(np.array([xx.ravel(), yy.ravel()]).T)
    Z = Z.reshape(xx.shape)

    # 결정 경계 그리기
    plt.contourf(xx, yy, Z, alpha=0.5, cmap=cmap)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

    # 클래스별 산점도
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y==cl, 0], y=X[y==cl, 1], color=cmap(idx),
                    marker=markers[idx], label=cl)
    # 테스트 데이터 강조
    if test_idx is not None:
        X_test = X[test_idx, :]
        plt.scatter(X_test[:, 0], X_test[:, 1], facecolors='none', edgecolors='k',
                    marker='o', s=80, label='test')

    plt.xlabel('꽃잎길이')
    plt.ylabel('꽃잎너비')
    plt.legend()
    plt.title(title)
    plt.show()

# train + test 데이터를 합쳐서 시각화
x_combined = np.vstack((x_train, x_test))
y_combined = np.hstack((y_train, y_test))

plot_decisionFunc(
    X=x_combined, y=y_combined, classifier=read_model,
    test_idx=range(len(x_train), len(x_combined)),
    title='Decision Tree 결정 경계'
)

# 10. 트리 시각화
from sklearn import tree
from io import StringIO
import pydotplus

dot_data = StringIO() # Graphviz용 데이터
tree.export_graphviz(
    read_model, out_file=dot_data, 
    feature_names=iris.feature_names[2:4]
)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('mytree.png')    # PNG 파일로 저장

from matplotlib.pyplot import imread
img = imread('mytree.png')
plt.imshow(img)
plt.axis('off')                  # 축 제거
plt.show()
