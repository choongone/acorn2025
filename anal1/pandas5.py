import pandas as pd
from pandas import Series, DataFrame
import numpy as np

# 대량의 데이터 파일을 읽는 경우 chunk 단위로 분리해 읽기 가능
# 1) 메모리 절약
# 2) 스트리밍 방식으로 순차적으로 처리 가능 (로그 분석)
# 3) 분산 처리(batch processing)
# 4) 다소 속도는 느릴 수 있음

import time
import matplotlib.pyplot as plt
plt.rc('font', family='Malgun Gothic')  # 한글 폰트 설정

n_rows = 10000
############ 대량의 데이터 생성을 위한 코드로, 생성후 더이상 필요하지 않아 주석처리###########
# data = {
#     'id': range(1, n_rows + 1),
#     'name': [f'Student_{i}' for i in range(1, n_rows + 1)],
#     'score1': np.random.randint(0, 101, size=n_rows),
#     'score2': np.random.randint(0, 101, size=n_rows)
# }
# df = pd.DataFrame(data)
# print('df head:\n', df.head(3))
# print('df tail:\n', df.tail(3))

# csv_path = 'anal2/students.csv'
# df.to_csv(csv_path, index=False)  # DataFrame을 CSV 파일로 저장
######### end ##############

# 작성된 csv 파일 사용: 전체 한 방에 처리
start_all = time.time()
df_all = pd.read_csv('anal2/students.csv')  # 전체 데이터를 한 번에 읽기
avg_score1 = df_all['score1'].mean()  # score1의 평균
avg_score2 = df_all['score2'].mean()  # score2의 평균
end_all = time.time()

print(f'전체 데이터 읽기 시간: {end_all - start_all:.4f}초')

# chunk 단위로 처리: 메모리 절약 및 속도 향상
chunk_size = 1000  # chunk 크기 설정
total_score1 = 0
total_score2 = 0
total_count = 0

start_chunk_total = time.time()
for i, chunk in enumerate(pd.read_csv('anal2/students.csv', chunksize=chunk_size)):
    start_chunk = time.time()
    # 청크 처리시 마다 첫번째 학생 정보 출력
    first_student = chunk.iloc[0]
    print(f'Chunk {i + 1} - 첫번째 학생 id = {first_student["id"]}, 이름 = {first_student["name"]}, '
          f'score1 = {first_student["score1"]}, score2 = {first_student["score2"]}')
    total_score1 += chunk['score1'].sum()  # score1 합계
    total_score2 += chunk['score2'].sum()  # score2 합계
    total_count += len(chunk)  # 총 학생 수
    end_chunk = time.time()
    print(f'Chunk {i + 1} 처리 시간: {end_chunk - start_chunk:.4f}초')

# 전체 청크 처리 시간
end_chunk_total = time.time()
print(f'전체 청크 처리 시간: {end_chunk_total - start_chunk_total:.4f}초')

# 최종 결과 출력
print(f'최종 결과:')
print(f'  전체 학생 수: {total_count}')
print(f'  score1 평균: {total_score1 / total_count:.4f}')
print(f'  score2 평균: {total_score2 / total_count:.4f}')

# 전체 한번에 처리한 시간과 청크 처리 시간 비교
print(f'전체 데이터 읽기 시간: {end_all - start_all:.4f}초')
print(f'청크 단위 처리 시간: {end_chunk_total - start_chunk_total:.4f}초') # 청크로 처리한 경우 시간이 더 오래 걸릴 수 있음

# 시각화
labels = ['전체 한번에 처리', '청크 단위 처리']
times = [end_all - start_all, end_chunk_total - start_chunk_total]
plt.figure(figsize=(6, 4))
bars = plt.bar(labels, times, color=['skyblue', 'yellow'])
for bar, time_val in zip(bars, times):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f'{time_val:.4f}초', 
             ha='center', va='bottom', fontsize=10)
plt.ylabel('처리시간 (초)')
plt.title('데이터 처리 시간 비교')
plt.grid(alpha=0.5)
plt.tight_layout()
plt.show()

# pandas로 파일 저장

items = { 
    'apple':{ 
        'count': 10,
        'price': 1500
    },
    'orange':{
        'count': 4,
        'price': 700
    }
}

df = pd.DataFrame(items)
print('df:\n', df)
# print('클립보드 \n:', df.to_clipboard())  # DataFrame을 클립보드에 복사
# print('html파일로 \n:', df.to_html())  # DataFrame을 HTML 파일로 저장
# print('json파일로 \n:', df.to_json()) # DataFrame을 JSON 파일로 저장
df.to_csv('anal2/result1.csv', index=False)  # DataFrame을 CSV 파일로 저장
df.to_csv('anal2/result2.csv', index=False, header=False)  # CSV 파일로 저장, 헤더 없이

data = df.T # DataFrame을 전치(transpose)하여 행과 열을 바꿈
data.to_csv('anal2/result3.csv', index=False)  # 전치된 DataFrame을 CSV 파일로 저장

# 엑셀 관련
df2 = pd.DataFrame({'name':['Alice','Bob','Oscar'], 'age':[24,26,33], 'city':['Seoul','Suwon','Incheon']})
print('df2:\n', df2)
# 저장
df2.to_excel('anal2/result4.xlsx', index=False, sheet_name='mySheet')  # DataFrame을 엑셀 파일로 저장
# 읽기
exdf = pd.ExcelFile('anal2/result4.xlsx')  # 엑셀 파일에서 DataFrame 읽기
print('df2_read:\n', exdf.sheet_names)  # 엑셀 파일의 시트 이름 출력
dfexcel = exdf.parse('mySheet')  # 'mySheet' 시트에서 DataFrame 읽기
print('dfexcel:\n', dfexcel)