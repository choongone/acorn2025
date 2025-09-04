# 회귀분석 문제 2) 

# testdata에 저장된 student.csv 파일을 이용하여 세 과목 점수에 대한 회귀분석 모델을 만든다. 
# 이 회귀문제 모델을 이용하여 아래의 문제를 해결하시오. 수학점수를 종속변수로 하자.
# - 국어 점수를 입력하면 수학 점수 예측
# - 국어, 영어 점수를 입력하면 수학 점수 예측
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
 








# 회귀분석 문제 3)    

# kaggle.com에서 carseats.csv 파일을 다운 받아 (https://github.com/pykwon에도 있음) Sales 변수에 영향을 주는 변수들을 선택하여 선형회귀분석을 실시한다.
# 변수 선택은 모델.summary() 함수를 활용하여 타당한 변수만 임의적으로 선택한다.
# 회귀분석모형의 적절성을 위한 조건도 체크하시오.
# 완성된 모델로 Sales를 예측.