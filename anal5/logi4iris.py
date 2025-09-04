from sklearn import datasets
import numpy as np
import pandas as pd 
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
plt.rc('font', family='malgun gothic')
plt.rcParams['axes.unicode_minus'] = False
from matplotlib.colors import ListedColormap
import pickle

from sklearn.linear_model import LogisticRegression     # 다중 클래스(종속변수=label=y=class) 지원 

iris = datasets.load_iris()
print(iris['data'])
print(np.corrcoef(iris.data[:,2], iris.data[:,3]))      # 0.96286
x = iris.data[:, [2, 3]]        # petal.length, petal.width만 작업에 참여. 타입: matrix
y = iris.target # 타입: vector
print(x[:3])
print(y[:3], set(y))

# traub / test split (7:3)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)     # (105, 2) (45, 2) (105,) (45,)

"""
# -------------------
# Scaling (데이터 표준화 - 최적화 과정에서 안전성, 수렴속도 향상, 오버플로우/언더플로우 방지 효과가 있음)
print(x_train[:3])
sc = StandardScaler()
sc.fit(x_train); sc.fit(x_test)
x_train = sc.transform(x_train)
x_test = sc.transform(x_test)   # 독립변수만 스케일링
print(x_train[:3])
# 스케일링 원복
inver_x_train = sc.inverse_transform(x_train)
print(inver_x_train[:3])
# -------------------
"""

# 분류 모델 생성
# C 속성 : L2규제 - 모델에 패널티 적용(tuning parameter 중 하나). 숫자 값을 조정해가며 최적의 분류 정확도를 확인 1.0, 10.0, 100.0 ... 값이 작을수록 더 강하나 정규화 규제를 지원함 
model = LogisticRegression(C=0.1, random_state=0, verbose=0)
print(model)
model.fit(x_train, y_train)     # supervised learning

# 분류 예측 - 모델 성능 파악용 
y_pred = model.predict(x_test)
print('예측값 : ', y_pred)      # 예측값 :  [2 1 0 2 0 2 0 1 1 1 2 1 1 1 1 0 1 1 0 0 2 1 0 0 2 0 0 1 1 0 2 1 0 2 2 1 0 2 1 1 2 0 2 0 0]
print('실제값 : ', y_test)      # 실제값 :  [2 1 0 2 0 2 0 1 1 1 2 1 1 1 1 0 1 1 0 0 2 1 0 0 2 0 0 1 1 0 2 1 0 2 2 1 0 1 1 1 2 0 2 0 0]

print('총 갯수:%d, 오류수:%d'%(len(y_test), (y_test != y_pred).sum()))  # 총 갯수:45, 오류수:1
print()

print('분류 정확도 확인1 : ')
print('%.5f'%accuracy_score(y_test, y_pred))

print('분류 정확도 확인2 : ')
con_mat = pd.crosstab(y_test, y_pred, rownames=['예측값'], colnames=['관측값'])
print(con_mat)
print((con_mat[0][0] + con_mat[1][1] + con_mat[2][2]) / len(y_test))

print('분류 정확도 확인3 : ')
print('test : ', model.score(x_test, y_test))
print('train : ', model.score(x_train, y_train))    # 두 개의 값 차이가 크면 과적합 의심 

# 모델 저장 
pickle.dump(model, open('logimodel.sav', 'wb'))
del model 

read_model = pickle.load(open('logimodel.sav', 'rb'))

# 새로운 값으로 예측 : petal.length, petal.width 만 참여 
# print(x_test[:3])
new_data = np.array([[5.1, 1.1],[1.1, 1.1],[6.1, 7.1]])
# 참고 : 만약 표준화한 데이터로 모델을 생성했다면 
# sc.fit(new_data); new_data = sc.transform(new_data)
new_pred = read_model.predict(new_data)     # 내부적으로 softmax가 출력한 값을 argmax로 처리 
print('에측 결과 : ', new_pred)
print(read_model.predict_proba(new_data))   # softmax가 출력한 값

# 시각화
def plot_decisionFunc(X, y, classifier, test_idx=None, resulution=0.02, title=''):
    # test_idx : test 샘플의 인덱스
    # resulution : 등고선 오차 간격
    markers = ('s','x','o','^','v')   # 마커(점) 모양 5개 정의함
    colors = ('r', 'b', 'lightgray', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])  # 색상팔레트를 이용
    # print(cmap.colors[0], cmap.colors[1])
    
    # surface(결정 경계) 만들기
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1  # 좌표 범위 지정
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    # 격자 좌표 생성
    xx, yy = np.meshgrid(np.arange(x1_min, x1_max, resulution), \
                         np.arange(x2_min, x2_max, resulution))
    
    # xx, yy를 1차원배열로 만든 후 전치한다. 이어 분류기로 클래스 예측값 Z얻기
    Z = classifier.predict(np.array([xx.ravel(), yy.ravel()]).T)
    Z = Z.reshape(xx.shape)  # 원래 배열(격자 모양)로 복원

    # 배경을 클래스별 색으로 채운 등고선 그리기
    plt.contourf(xx, yy, Z, alpha=0.5, cmap=cmap)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

    X_test = X[test_idx, :]
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y==cl, 0], y=X[y==cl, 1], color=cmap(idx), \
                    marker=markers[idx], label=cl)
    if test_idx:
        X_test = X[test_idx, :]
        plt.scatter(x=X[:, 0], y=X[:, 1], color=[], \
                    marker='o', linewidths=1, s=80, label='test')
    plt.xlabel('꽃잎길이')
    plt.ylabel('꽃잎너비')
    plt.legend()
    plt.title(title)
    plt.show()

# train과 test 모두를 한 화면에 보여주기 위한 작업 진행
# train과 test 자료 수직 결합(위 아래로 이어 붙임 - 큰행렬 X 작성)
x_combined_std = np.vstack((x_train, x_test))   # feature
# 좌우로 이어 붙여 하나의 큰 레이블 벡터 y 만들기
y_combined = np.hstack((y_train, y_test))    # label
plot_decisionFunc(X=x_combined_std, y=y_combined, classifier=read_model, \
                  test_idx = range(100, 150), title='scikit-learn 제공')












