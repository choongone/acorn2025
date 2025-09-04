# numpy 기본 기능 
import numpy as np 

ss = ['tom','james','oscar', 5] # list는 온갖 형태를 다 넣을 수 있음    # 배열은 타입이 일정해야됨(동일 데이터만 저장 가능)
print(ss,type(ss))
ss2 = np.array(ss) # int < float < string
print(ss2, type(ss2))

# 메모리 비교
li = list(range(1,10))
print(li)
print(id(li[0]), ' ', id(li[1]))
print(li*10) # li라는 것을 10번 생성해라
print('^' * 10)

for i in li:
    print(i * 10, end = ' ')
print()
print([i * 10 for i in li])

print('---')

num_arr = np.array(li)
print(id(num_arr[0]), ' ', id(num_arr[1]))
print(num_arr * 10)

print() # 라인스킵
a = np.array([1,2,0,'3']) 
print(a, type(a), a.dtype, a.shape, a.ndim, a.size)
print(a[0],a[1])
# int -> float -> complex -> str

b = np.array([[1,2,3],[4,5,6]])
print(b.shape, ' ', b[0], ' ', b[[0]])
print(b[0, 0], ' ', b[1, 2])
print()

c = np.zeros((2,2))
print(c)

d = np.ones((2,2))
print(d)

e = np.full((2,2), fill_value=7)
print(e)

f = np.eye(3)
print(f)
print()

print(np.mean(np.random.rand(5000))) # 0 ~ 1 사이의 난수(균등분포)
print(np.mean(np.random.randn(5000))) # 정규분포
np.random.seed(42)
print(np.random.randn(2,3))

print('\n배열 인덱싱 ------------')
a = np.array([1,2,3,4,5])
print(a) # 인덱싱
print(a[1]) # 슬라이싱
print(a[1:])
print(a[1:5:2])
print(a[-2:])
print()

a = np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12]])
print(a)
print(a[:])
print(a[1:])
print(a[1:, 0:2])
print(a[0], ' ', a[0][0], ' ', a[[0]])
print()

aa = np.array((1,2,3))
print(aa)
bb = aa[1:3] # 서브 배열(논리적)
print(bb, ' ', bb[0])
bb[0] = 33
print(bb)
print(aa)
cc = aa[1:3].copy() # 복사본 생성
print(cc)
cc[0] = 55
print(cc)
print(aa)
print('---------')
a = np.array([[1,2,3],[4,5,6],[7,8,9]])
r1 = a[1, :]  # 1차원 배열
r2 = a[1:2, :] # 2차원 배열
print(r1, r1.shape)
print(r2, r2.shape)
print()
c1 = a[:, 1]  # 전체행 1열 슬라이싱
c2 = a[:, 1:2]  # 전체행 1열 슬라이싱
print(c1, c1.shape) # [2 5 8] (3,)
print(c2, c2.shape) # [[2] [5] [8]] (3, 1)
print()

print(a)
bool_idx = (a >= 5)
print(bool_idx)
print(a[bool_idx])




