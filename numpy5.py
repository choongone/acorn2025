# 배열에 행 또는 열 추가 
import numpy as np

aa = np.eye(3)
print('aa: \n', aa)
bb = np.c_[aa, aa[2]]
print(bb)
cc = np.r_[aa, [aa[2]]]
print(cc)

# reshape
a = np.array([1,2,3])
print('np.c_:\n', np.c_[a]) # np.c_ (r도 가능)로 구조를 바꿀 수 있음 
a.reshape(3,1)
print(a)

print('--append, insert, delete--')
print(a)
b = np.append(a, [4, 5])
print(b)
c = np.insert(a, 2, [6, 7]) # a에 2번째에 6,7을 삽입
print(c)
# d = np.delete(a, 1)
# d = np.delete(a, [1])
d = np.delete(a, 1) # a에 1번째를 삭제 
print(d)
print()

# 2차원
aa = np.arange(1, 10).reshape(3,3) # 1~9까지 3행 3열로 
print(aa)
print(np.insert(aa, 1, 99)) # 삽입 후 차원 축소
print(np.insert(aa, 1, 99, axis=0)) # 1 행 기준에서 99로 채우기 
print(np.insert(aa, 1, 99, axis=1)) # 1 열 기준에서 99로 채우기 

print(aa)
bb = np.arange(10,19).reshape(3,3)
print(bb)
cc = np.append(aa, bb) # 추가 후 차원 축소 
print(cc)
cc = np.append(aa, bb, axis=0)
print(cc)
cc = np.append(aa, bb, axis=1)
print(cc)

print("np.append 연습")
print(np.append(aa, 88))
print(np.append(aa, [[88,88,88]], axis=0))
print(np.append(aa, [[88],[88],[88]], axis=1))

print()
print(np.delete(aa, 1)) # 삭제 후 차원 축소
print(np.delete(aa, 1, axis=0))
print(np.delete(aa, 1, axis=1))

# 조건 연산 where(조건, 참, 거짓)
x = np.array([1,2,3])
y = np.array([4,5,6])
condData = np.array([True, False, True])
result = np.where(condData, x, y)
print(result)

aa = np.where(x >= 2)
print(aa) # (array([1, 2]),)  index
print(x[aa]) # index 값을 출력
print(np.where(x >= 2, 'T', 'F'))
print(np.where(x >= 2, x, x + 100)) # 참일때는 x로, 거짓일때는 수식으로 나오도록

bb = np.random.randn(4, 4) # 정규분포(가우시안분포 - 중심극한정리)를 따르는 

print(bb)
print(np.where(bb > 0, 7, bb)) # bb가 0보다 크면 7을 출력, 아니면 bb를 출력 

print('배열 결합/분할')
kbs = np.concatenate([x, y])
print(kbs)

x1, x2 = np.split(kbs, 2)
print(x1)
print(x2)
print()
a = np.arange(1, 17).reshape(4,4)
print(a)
x1, x2 = np.hsplit(kbs, 2)
print(x1)
print(x2)
x1, x2 = np.vsplit(a, 2)
print(x1)
print(x2)

print('복원, 비복원 추출') 
datas = np.array([1,2,3,4,5,6,7])

# 복원 추출
for _ in range(5):
    print(datas[np.random.randint(0, len(datas) - 1)], end = ' ')
print()

# 비복원 추출 전용 - sample()
import random
print(random.sample(datas.tolist(), 5))

print('------------')

# 추출 햠수 : choice()
# 복원 추출
print(np.random.choice(range(1, 46), 6)) # [ 9 31 9 36 5 1]
# 비복원 추출
print(np.random.choice(range(1,46), 6, replace=False))


# 가중치를 부여한 랜덤 추출 
ar = 'air book cat d e f god'
ar = ar.split(' ')
print(ar)
print(np.random.choice(ar, 3, p=[0.1,0.1,0.1,0.1,0.1,0.1,0.4])) # air book cat d e f god 각각 나올 확률 부여 
