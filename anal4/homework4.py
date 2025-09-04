# 원격 DB의 jikwon 테이블에서 근무년수에 대한 연봉을 이용하여 회귀분석 모델을 작성하시오.
# 장고로 작성한 웹에서 근무년수를 입력하면 예상 연봉이 나올 수 있도록 프로그래밍 하시오.
# LinearRegression 사용. Ajax 처리!!!      참고: Ajax 처리가 힘들면 그냥 submit()을 해도 됩니다.

# --- html문서 ---
# <script src="http://code.jquery.com/jquery-latest.min.js"></script>
# <script>
# function 함수명(){
#   ...
#   $.ajax({
#      url:       ,
#      type:"post",
#      data:     ,
#      dataType:"json",
#      success:function(data){
#      }
#   });
# }
# </script>
# --- views.py ---
# from django.http.response import HttpResponse, JsonResponse
# ...
# 함수에서 처리 후
#   # json으로 전송하기
#   return JsonResponse({'키':값})
#   또는 
#   return HttpResponse(json.dumps({'키':값}, content_type='application/json')
# html 파일 레이아웃
# ________________________________
#   근무년수 :  ***     확인버튼
#    예상연봉액 : --------
#    설명력 : **%
#    직급별 연봉평균 
#    ---
#    ---
#    ...
# _________________________________
# 참고 : Ajx 처리를 할 때 임의의 함수에서 csrf를 적용하고 싶지 않은 경우에는  아래와 같이 코드를 적어 주면 된다.
# @csrf_exempt
# def 함수명:
#     ...
"""
import pandas as pd
import MySQLdb
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import datetime

@csrf_exempt
def predict_salary(request):
    try:
        # 1. 원격 DB에서 데이터 가져오기
        conn = MySQLdb.connect(
            host='acorn',
            user='your_db_user',
            passwd='1234',
            db='mydb',
            charset='utf8'
        )
        cursor = conn.cursor()
        
        sql = "SELECT jikwon_ibsail, jikwon_pay, jikwon_jik FROM jikwon"
        df = pd.read_sql(sql, conn)
        
        # 날짜를 근무년수로 변환
        df['jikwon_ibsail'] = pd.to_datetime(df['jikwon_ibsail'])
        df['work_years'] = (datetime.datetime.now() - df['jikwon_ibsail']).dt.days // 365

        # 2. 회귀 모델 학습
        X = df[['work_years']]
        y = df['jikwon_pay']
        
        model = LinearRegression()
        model.fit(X, y)
        
        # 3. 모델 성능 평가 (설명력 계산)
        y_pred = model.predict(X)
        r_squared = r2_score(y, y_pred)
        
        # 4. 입력받은 근무년수로 연봉 예측
        if request.method == 'POST':
            work_years_input = int(request.POST.get('work_years', 0))
            predicted_salary = model.predict([[work_years_input]])[0]
            
            # 5. 직급별 연봉 평균 계산
            avg_salaries = df.groupby('jikwon_jik')['jikwon_pay'].mean().to_dict()
            
            conn.close()

            # 6. JSON 응답
            return JsonResponse({
                'predicted_salary': int(predicted_salary),
                'r_squared': round(r_squared * 100, 2),
                'avg_salaries': avg_salaries
            })
            
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)
"""












# 회귀분석 문제 5) 
# Kaggle 지원 dataset으로 회귀분석 모델(LinearRegression)을 작성하시오.
# testdata 폴더 : Consumo_cerveja.csv
# Beer Consumption - Sao Paulo : 브라질 상파울루 지역 대학생 그룹파티에서 맥주 소모량 dataset
# feature : Temperatura Media (C) : 평균 기온(C)
#             Precipitacao (mm) : 강수(mm)
# label : Consumo de cerveja (litros) - 맥주 소비량(리터) 를 예측하시오
# 조건 : NaN이 있는 경우 삭제!
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('Consumo_cerveja.csv')

df.columns = ['날짜', '평균기온', '최저기온', '최고기온', '강수량', '주말여부', '맥주소비량']
df = df.drop(['날짜', '최저기온', '최고기온', '주말여부'], axis=1)
# 데이터에 있는 ','를 '.'으로 변환하고 숫자형으로 변경
for col in ['평균기온', '강수량', '맥주소비량']:
    df[col] = df[col].str.replace(',', '.', regex=True).astype(float)
# NaN 값이 있는 행 삭제 (조건)
df = df.dropna()
# feature (X)와 label (y) 정의
features = ['평균기온', '강수량']
label = ['맥주소비량']
X = df[features]
y = df[label]
# 학습 데이터와 테스트 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# LinearRegression 모델 생성 및 학습
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"MSE (평균 제곱 오차): {mse:.2f}")
print(f"R-squared (결정 계수): {r2:.2f}")
# 회귀 계수(기울기)와 절편 출력
print("\n회귀 계수:", model.coef_)
print("절편:", model.intercept_)
# 특정 값으로 예측 예시
# 평균 기온 25°C, 강수량 10mm일 때 맥주 소비량 예측
new_data = np.array([[25, 10]])
predicted_consumption = model.predict(new_data)
print(f"\n평균 기온 25°C, 강수량 10mm일 때 예상 맥주 소비량: {predicted_consumption[0][0]:.2f} 리터")

























