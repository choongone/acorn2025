from sklearn.datasets import load_iris                  # 붓꽃 데이터셋 로드
from sklearn.tree import DecisionTreeClassifier         # 의사결정트리 모델
from sklearn.metrics import accuracy_score             # 정확도 평가 함수
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, cross_val_score
import numpy as np                                     # 수치 연산용 라이브러리
import pandas as pd

# 데이터셋 불러오기
iris = load_iris()
print(iris.keys())                                     # 데이터셋의 구성 확인 (data, target 등)
train_data = iris.data                                 # 특성(feature) 데이터
train_label = iris.target                              # 라벨(target) 데이터
print(train_data[:3])                                  # 처음 3개 데이터 확인
print(train_label[:3])                                 # 처음 3개 라벨 확인

# 분류 모델 초기화 및 학습
dt_clf = DecisionTreeClassifier()                     # 기본 의사결정트리 모델 생성
print(dt_clf)
dt_clf.fit(train_data, train_label)                  # 전체 데이터로 학습 (과적합 가능)
pred = dt_clf.predict(train_data)                     # 학습 데이터로 예측
print('예측값 : ', pred)
print('실제값 : ', train_label)
print('분류 정확도 : ', accuracy_score(train_label, pred)) 
# 전체 데이터 학습 시 정확도가 매우 높을 수 있음 → 과적합 가능

print('과적합 방지 방법 1 : train/test 로 분리')
# train/test split : 데이터를 학습용과 테스트용으로 분리
x_train, x_test, y_trian, y_test = train_test_split(
    iris.data, iris.target, test_size=0.3, shuffle=True, random_state=121
)
print(x_train.shape, x_test.shape, y_trian.shape, y_test.shape)  # 데이터 크기 확인
dt_clf.fit(x_train, y_trian)                    # 학습용 데이터로 학습
pred2 = dt_clf.predict(x_test)                  # 테스트용 데이터로 예측
print('예측값 : ', pred2)
print('실제값 : ', y_test)
print('분류 정확도 : ', accuracy_score(y_test, pred2)) 
# train/test 분리 시 과적합 감소, 모델 일반화 성능 확인 가능

print('\n과적합 방지 방법 2 : 교차검증(cross validation)')
# K-fold 교차 검증 : 데이터를 k개의 fold로 나누어 학습/검증 반복
features = iris.data
labels = iris.target
dt_clf = DecisionTreeClassifier(criterion='entropy', random_state=123)  # 정보 이득 기준
kfold = KFold(n_splits=5)  # 5-Fold 교차검증
cv_acc = []                 # 반복마다 정확도 저장
print('iris shape : ', features.shape) 

n_iter = 0
for train_index, test_index in kfold.split(features):
    # kfold.split는 학습용/검증용 인덱스를 반환
    xtrain, xtest = features[train_index], features[test_index]  # 학습/검증 데이터 분리
    ytrian, ytest = labels[train_index], labels[test_index]      # 학습/검증 라벨 분리

    # 학습 및 검증
    dt_clf.fit(xtrain, ytrian)           # 학습
    pred = dt_clf.predict(xtest)         # 검증 데이터 예측
    n_iter += 1

    # 반복 시 정확도 측정
    acc = np.round(accuracy_score(ytest, pred), 3)
    train_size = xtrain.shape[0]
    test_size = xtest.shape[0]
    print('반복수 : {0}, 교차검증 정확도 : {1}, 학습데이터 수 : {2}, 검증데이터 수 : {3}'
          .format(n_iter, acc, train_size, test_size))
    print('반복수 : {}, 검증인덱스 : {}'.format(n_iter, test_index))
    cv_acc.append(acc)  # 반복마다 정확도 저장

print('평균 검증 정확도 : ', np.mean(cv_acc))  # K-fold 평균 검증 정확도 : 0.9199

print('\n----------------------\n')

# StratifiedGroupKFold : 불균형한 분포를 가진 데이터 집합을 위한 kfold 방식
# 편향된 분포를 가진 데이터
# ex) 대출사기, 이메일(스팸), 강우량, 코로나 검사결과
from sklearn.model_selection import StratifiedKFold

skfold = StratifiedKFold(n_splits=5)  
cv_acc = []                
n_iter = 0

for train_index, test_index in skfold.split(features, labels):
    xtrain, xtest = features[train_index], features[test_index]  
    ytrian, ytest = labels[train_index], labels[test_index]      

    # 학습 및 검증
    dt_clf.fit(xtrain, ytrian)         
    pred = dt_clf.predict(xtest)        
    n_iter += 1

    # 반복 시 정확도 측정
    acc = np.round(accuracy_score(ytest, pred), 3)
    train_size = xtrain.shape[0]
    test_size = xtest.shape[0]
    print('반복수 : {0}, 교차검증 정확도 : {1}, 학습데이터 수 : {2}, 검증데이터 수 : {3}'
          .format(n_iter, acc, train_size, test_size))
    print('반복수 : {}, 검증인덱스 : {}'.format(n_iter, test_index))
    cv_acc.append(acc)  # 반복마다 정확도 저장

print('평균 검증 정확도 : ', np.mean(cv_acc))

print('교차검증 함수 처리 ---')
data = iris.data
label = iris.target
score = cross_val_score(dt_clf, data, label, scoring = 'accuracy', cv = 5)
print('교차 검증별 정확도 : ', np.round(score, 2))
print('평균 검증 정확도 : ', np.round(np.mean(score), 2))

print('\n과적합 방지 방법 3 : GridsearchCV - 최적의 파라미터를 제공')
parameters = {'max_depth' : [1,2,3], 'min_samples_split' : [2,3]} # dict type
grid_dtree = GridSearchCV(dt_clf, param_grid = parameters, cv = 3, refit = True)
grid_dtree.fit(x_train, y_trian) # 자동으로 복수의 내부 모형 생성, 실행하며 최적의 파라미터 탐색

scoreDF = pd.DataFrame(grid_dtree.cv_results_)
# pd.set_option('display.max_columns', None)
print(scoreDF)
print('best parameter : ', grid_dtree.best_params_) 
print('best accuracy : ', grid_dtree.best_score_)   

# 최적의 parameter를 탑재한 모델이 제공
estimeter = grid_dtree.best_estimator_
pred = estimeter.predict(x_test)
print('예측값 : ', pred)
print('테스트 데이터 정확도 : ', accuracy_score(y_test, pred))
