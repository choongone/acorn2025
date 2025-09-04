# 배열 연산

import numpy as np

x = np.array([[1,2],[3,4]], dtype=np.float64)
y = np.arange(5, 9).reshape(2,2)   # 파이썬의 range와 같은 의미 
y = y.astype(np.float64)
print(x, x.astype, x.dtype)
print(y, y.astype, y.dtype)

# 요소별 합 
print(x + y) # python의 산술 연산자 사용
print(np.add(x, y)) # numpy의 함수 
# python의 연산속도가 numpy의 연산속도에 비해 느림 
# np.subtract, np.multiply, np.divide
import time 
big_arr = np.random.rand(1000000)

start = time.time()
sum(big_arr)     # python 함수
end = time.time()
print(f"sum():{end - start:.6f}sec")

start = time.time()
np.sum(big_arr)  # numpy 함수
end = time.time()
print(f"np.sum():{end - start:.6f}sec")

# 요소별 곱 
print(x)
print(y)
print(x * y)
print(np.multiply(x, y))

print(x.dot(y)) # 내적 연산 
print()
# 내적은 머신러닝에서 절대적임 
# 내적을 알려면 벡터를 알아야함 

v = np.array([9, 10])
w = np.array([11,12])
print(v * w)
print(v.dot(w)) # (9*11)+(10*12) # 값이 스칼라로 나옴 # 두 벡터의 내적의 크기
print(np.dot(v, w)) # nonpy의 계산 방법이 속도가 더 빠름
print(np.dot(x, v))
print()

print('유용한 함수 ----------')
print(x)
print(np.sum(x, axis = 0))  # 열 단위 연산
print(np.sum(x, axis = 1))  # 행 단위 연산

print(np.min(x), ' ', np.max(x))
print(np.argmax(x), ' ', np.argmax(x)) # 인덱스 반환
print(np.cumsum(x)) # 누적 값
print(np.cumprod(x)) # 누적 곱
print()

names = np.array(['tom','james','oscar', 'tom', 'oscar', 'abc']) 
names2 = np.array(['tom', 'page', 'john'])
print(np.unique(names)) # 안겹치는 이름만 나오도록 # 중복을 허용하지 않음 
print(np.intersect1d(names, names2, assume_unique=True)) # 교집합 
print(np.union1d(names, names2)) 
print('\n전치(Transpose)')  # 행과 열을 바꿔줌
print(x)
print(x.T)
arr = np.arange(1,16).reshape((3,5)) # 3행, 5열로 보여줌
print(arr)
print(arr.T)
print(np.dot(arr.T, arr))

print(arr.flatten()) # 차원 축소(2차원에서 1차원으로)
print(arr.ravel())






