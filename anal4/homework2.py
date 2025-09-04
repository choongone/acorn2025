# 회귀분석 문제 1) scipy.stats.linregress() <= 꼭 하기 : 심심하면 해보기 => statsmodels ols(), LinearRegression 사용
# 나이에 따라서 지상파와 종편 프로를 좋아하는 사람들의 하루 평균 시청 시간과 운동량에 대한 데이터는 아래와 같다.
#  - 지상파 시청 시간을 입력하면 어느 정도의 운동 시간을 갖게 되는지 회귀분석 모델을 작성한 후에 예측하시오.
#  - 지상파 시청 시간을 입력하면 어느 정도의 종편 시청 시간을 갖게 되는지 회귀분석 모델을 작성한 후에 예측하시오. (참고로 결측치는 해당 칼럼의 평균 값을 사용하기로 한다. 이상치가 있는 행은 제거. 운동 10시간 초과는 이상치로 한다.)  

# 구분,지상파,종편,운동
#   1,0.9,0.7,4.2
#   2,1.2,1.0,3.8
#   3,1.2,1.3,3.5
#   4,1.9,2.0,4.0
#   5,3.3,3.9,2.5
#   6,4.1,3.9,2.0
#   7,5.8,4.1,1.3
#   8,2.8,2.1,2.4
#   9,3.8,3.1,1.3
#   10,4.8,3.1,35.0
#   11,NaN,3.5,4.0
#   12,0.9,0.7,4.2
#   13,3.0,2.0,1.8
#   14,2.2,1.5,3.5
#   15,2.0,2.0,3.5

# data1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
# data2 = [0.9, 1.2, 1.2, 1.9, 3.3, 4.1, 5.8, 2.8, 3.8, 4.8, 2.6, 0.9, 3.0, 2.2, 2.0]
# data3 = [0.7, 1.0, 1.3, 2.0, 3.9, 3.9, 4.1, 2.1, 3.1, 3.1, 3.5, 0.7, 2.0, 1.5, 2.0]
# data4 = [4.2, 3.8, 3.5, 4.0, 2.5, 2.0, 1.3, 2.4, 1.3, 10, 4.0, 4.2, 1.8, 3.5, 3.5]

# from scipy import stats
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# print(np.mean(data2))  # 2.7

# x = data2
# y = data4
# z = data3

# model = stats.linregress(x, y)
# print(model)    # slope=np.float64(-0.07761492338441042), intercept=np.float64(3.676226959804575), rvalue=np.float64(-0.054523003287014336), 
#                 # pvalue=np.float64(0.8469673868211391), stderr=np.float64(0.3942278480414499), intercept_stderr=np.float64(1.2016730458935707
# print('기울기 : ', model.slope)                     
# print('절편 : ', model.intercept)                  
# print('R² - 결정계수(설명력) : ', model.rvalue)      
# print('p-value : ', model.pvalue)                  
# print('표준오차 : ', model.stderr)      

# data1_array = np.array(data1)
# data2_array = np.array(data2)
# data3_array = np.array(data3)
# data4_array = np.array(data4)

# plt.scatter(x, y)
# plt.plot(data2_array, model.slope * data2_array + model.intercept)
# plt.show()

# print('운동시간 예측 : ', model.slope * data2_array + model.intercept)


# # 2 
# model2 = stats.linregress(x, z)
# print(model2)   # LinregressResult(slope=np.float64(0.7051965356429045), intercept=np.float64(0.4226360204308248), rvalue=np.float64(0.8676837366702655), 
#                 # pvalue=np.float64(2.7741534151859616e-05), stderr=np.float64(0.11205605246254693), intercept_stderr=np.float64(0.3415657684824961))

# print('기울기 : ', model2.slope)                     
# print('절편 : ', model2.intercept)                  
# print('R² - 결정계수(설명력) : ', model2.rvalue)      
# print('p-value : ', model2.pvalue)                  
# print('표준오차 : ', model2.stderr)      

# plt.scatter(x, z)
# plt.plot(data2_array, model2.slope * data2_array + model2.intercept)
# plt.show()

# print('운동시간 예측 : ', model2.slope * data2_array + model2.intercept)


# 회귀분석 문제 1) scipy.stats.linregress() <= 꼭 하기 : 심심하면 해보기 => statsmodels ols(), LinearRegression 사용
import pandas as pd
import numpy as np
from scipy.stats import linregress
import matplotlib.pyplot as plt
import seaborn as sns

# 구분,지상파,종편,운동
# 1,0.9,0.7,4.2
# 2,1.2,1.0,3.8
# 3,1.2,1.3,3.5
# 4,1.9,2.0,4.0
# 5,3.3,3.9,2.5
# 6,4.1,3.9,2.0
# 7,5.8,4.1,1.3
# 8,2.8,2.1,2.4
# 9,3.8,3.1,1.3
# 10,4.8,3.1,35.0
# 11,NaN,3.5,4.0
# 12,0.9,0.7,4.2
# 13,3.0,2.0,1.8
# 14,2.2,1.5,3.5
# 15,2.0,2.0,3.5

# 나이에 따라서 지상파와 종편 프로를 좋아하는 사람들의 하루 평균 시청 시간과 운동량에 대한 데이터
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

data = {
    '구분': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    '지상파': [0.9, 1.2, 1.2, 1.9, 3.3, 4.1, 5.8, 2.8, 3.8, 4.8, np.nan, 0.9, 3.0, 2.2, 2.0],
    '종편': [0.7, 1.0, 1.3, 2.0, 3.9, 3.9, 4.1, 2.1, 3.1, 3.1, 3.5, 0.7, 2.0, 1.5, 2.0],
    '운동': [4.2, 3.8, 3.5, 4.0, 2.5, 2.0, 1.3, 2.4, 1.3, 35.0, 4.0, 4.2, 1.8, 3.5, 3.5]
}

df = pd.DataFrame(data)
print("원본 데이터:")
print(df)
print("\n" + "="*50)

# 이상치 제거 (운동 12시간 초과)
print("데이터 전처리")
outlier_rows = df[df['운동'] > 12]
print(f"이상치 발견: {len(outlier_rows)}개 행")
if len(outlier_rows) > 0:
    print(f"제거할 행: {outlier_rows['구분'].values}번 (운동시간: {outlier_rows['운동'].values})")

df_clean = df[df['운동'] <= 12].copy()
print(f"데이터 개수: {len(df)} → {len(df_clean)}개")

# 결측치 처리 - 평균값으로 대체
지상파_평균 = df_clean['지상파'].mean()
종편_평균 = df_clean['종편'].mean()

print(f"\n결측치 처리:")
print(f"지상파 평균: {지상파_평균:.2f}")
print(f"종편 평균: {종편_평균:.2f}")

df_clean['지상파'] = df_clean['지상파'].fillna(df_clean['지상파'].mean())
df_clean['종편'] = df_clean['종편'].fillna(df_clean['종편'].mean())

# ========================================================================================
# 모델 1: 지상파 시청 시간 → 운동 시간 예측
# ========================================================================================
print("\n" + "="*60)
print("모델 1: 지상파 시청시간 → 운동시간 예측")
print("="*60)

X1 = df_clean['지상파'].values
y1 = df_clean['운동'].values

slope1, intercept1, r_value1, p_value1, std_err1 = linregress(X1, y1)

print(f"회귀분석 결과 (scipy.stats.linregress):")
print(f"   기울기(slope): {slope1:.4f}") #기울기는 0.0000 이런식으로 나옴
print(f"   절편(intercept): {intercept1:.4f}") #절편은 0이 아니라 0.0000 이런식으로 나옴
print(f"   상관계수(r): {r_value1:.4f}") #상관계수는 0.0000 이런식으로 나옴
print(f"   결정계수(R²): {r_value1**2:.4f}") #결정계수는 0.0000 이런식으로 나옴
print(f"   p-value: {p_value1:.6f}") 

print(f"\n회귀방정식:")
print(f"   운동시간 = {slope1:.4f} × 지상파시청시간 + {intercept1:.4f}")

# 예측 함수
def predict_exercise(terrestrial_hours):
    return slope1 * terrestrial_hours + intercept1

# ========================================================================================
# 모델 2: 지상파 시청 시간 → 종편 시청 시간 예측
# ========================================================================================
print("\n" + "="*60)
print("모델 2: 지상파 시청시간 → 종편시청시간 예측")
print("="*60)

X2 = df_clean['지상파'].values
y2 = df_clean['종편'].values

slope2, intercept2, r_value2, p_value2, std_err2 = linregress(X2, y2)

print(f"회귀분석 결과 (scipy.stats.linregress):")
print(f"   기울기(slope): {slope2:.4f}")
print(f"   절편(intercept): {intercept2:.4f}")
print(f"   상관계수(r): {r_value2:.4f}")
print(f"   결정계수(R²): {r_value2**2:.4f}")
print(f"   p-value: {p_value2:.6f}")

#회귀 방정식 

print(f"   종편시청시간 = {slope2:.4f} × 지상파시청시간 + {intercept2:.4f}")

# 예측 함수
def predict_cable(terrestrial_hours):
    return slope2 * terrestrial_hours + intercept2

# ========================================================================================
# 예측 결과 출력
# ========================================================================================
print("\n" + "="*60)
print("예측 결과")
print("="*60)

test_values = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
print("지상파시간 | 예상운동시간 | 예상종편시간")
print("-" * 38)

for tv_time in test_values:
    exercise_pred = predict_exercise(tv_time)
    cable_pred = predict_cable(tv_time)
    print(f"{tv_time:4.1f}시간   | {exercise_pred:6.2f}시간   | {cable_pred:6.2f}시간")

# ========================================================================================
# 시각화
# ========================================================================================

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# 모델 1 시각화 (지상파 → 운동)
axes[0].scatter(X1, y1, alpha=0.7, color='blue', s=80)
x_line1 = np.linspace(X1.min(), X1.max(), 100)
y_line1 = slope1 * x_line1 + intercept1
axes[0].plot(x_line1, y_line1, 'r-', linewidth=2)
axes[0].set_xlabel('지상파 시청 시간 (hours)', fontsize=11)
axes[0].set_ylabel('운동 시간 (hours)', fontsize=11)
axes[0].set_title(f'모델 1: 지상파 → 운동\nR² = {r_value1**2:.3f}, r = {r_value1:.3f}', fontsize=12)
axes[0].grid(True, alpha=0.3)

# 모델 2 시각화 (지상파 → 종편)
axes[1].scatter(X2, y2, alpha=0.7, color='green', s=80)
x_line2 = np.linspace(X2.min(), X2.max(), 100)
y_line2 = slope2 * x_line2 + intercept2
axes[1].plot(x_line2, y_line2, 'r-', linewidth=2)
axes[1].set_xlabel('지상파 시청 시간 (hours)', fontsize=11)
axes[1].set_ylabel('종편 시청 시간 (hours)', fontsize=11)
axes[1].set_title(f'모델 2: 지상파 → 종편\nR² = {r_value2**2:.3f}, r = {r_value2:.3f}', fontsize=12)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# ========================================================================================
# 분석 결과 해석
# ========================================================================================

print("\n" + "="*60)
print("분석 결과 해석")
print("="*60)

def interpret_correlation(r):
    if abs(r) >= 0.7:
        return "강한"
    elif abs(r) >= 0.3:
        return "중간"
    else:
        return "약한"

def interpret_direction(r):
    return "양의" if r > 0 else "음의"

print(f"데이터 전처리:")
print(f"   - 원본 데이터: {len(df)}개")
print(f"   - 이상치 제거: {len(df) - len(df_clean)}개 (운동 > 10시간)")
print(f"   - 최종 데이터: {len(df_clean)}개")

print(f"\n모델 1 (지상파 → 운동):")
print(f"   - 회귀식: 운동 = {slope1:.3f} × 지상파 + {intercept1:.3f}")
print(f"   - 상관관계: {interpret_correlation(r_value1)} {interpret_direction(r_value1)} 상관 (r = {r_value1:.3f})")
print(f"   - 설명력: {r_value1**2*100:.1f}% (R² = {r_value1**2:.3f})")
print(f"   - 해석: 지상파 1시간 증가 → 운동시간 {abs(slope1):.2f}시간 {'감소' if slope1 < 0 else '증가'}")

print(f"\n모델 2 (지상파 → 종편):")
print(f"   - 회귀식: 종편 = {slope2:.3f} × 지상파 + {intercept2:.3f}")
print(f"   - 상관관계: {interpret_correlation(r_value2)} {interpret_direction(r_value2)} 상관 (r = {r_value2:.3f})")
print(f"   - 설명력: {r_value2**2*100:.1f}% (R² = {r_value2**2:.3f})")
print(f"   - 해석: 지상파 1시간 증가 → 종편시청 {slope2:.2f}시간 {'감소' if slope2 < 0 else '증가'}")

print(f"\nscipy.stats.linregress() 사용한 회귀분석 완료!")









