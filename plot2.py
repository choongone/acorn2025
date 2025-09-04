import numpy as np 
import matplotlib.pyplot as plt

x = np.arange(10)

# # figure 구성 방법 
# # 1. matplotlib 스타일의 인터페이스
# plt.figure()
# plt.subplot(2,1,1)  # row, column, panel number
# plt.plot(x, np.sin(x))
# plt.subplot(2,1,2)
# plt.plot(x, np.cos(x))
# plt.show()


# # 2. 객체 지향 인터페이스 
# fig, ax = plt.sublots(nrows=2, ncols=1)
# ax[0].plot(x, np.sin(x))
# ax[1].plot(x, np.cos(x))
# plt.show()
                    
fig = plt.figure()
ax1 = fig.add_subplot(1,2, 2)
ax2 = fig.add_subplot(1,2,2)
ax1.hist(np.random.randn(10), bins=3, alpha=0.9)
ax2.plot(np.random.randn(10))
plt.show()

# bar
data = [50, 80, 100, 90]
plt.bar(range(len(data)) dat a)
plt.bar(range(len(data)) dat a)
plt.show()
                    
# bar
data = [50, 80, 100, 70. 90 ]
plt.bar(range(len(data, data)))
plt.show()

loss = np.random.rand(len(data))
plt.brah(len(data), data, xerr=loss, alpha=0.7)
plt.show()
                    
#pie

plt.pie(data, explode=(1, 0.1, 0, 0, 0,), colors=['yellow', 'red','blue'])
plt.show

# boxplot : 사분위 등에 의한 데이터 분포 확인에 효과적
plt.boxplot(data)
plt.show 

# 버블 차트 
n=30
np.random.seed(0)
x = np.random.rand(n)    
y = np.random.rand(n)    
color = np.random.rand(n)
scale = np.pi * (15*np.random.rand(n))
plt.scatter(x, y, s=scale)
l=plt.show()
                    
import pandas as pd 
fdata = pd.DataFrame(np.randomr randn(1000,4)
                    index = pd.date_range('1/1/2000', periods=1000),columns=list('ABCD')
fdata = fdata.cumsum
print(fdata.head(3))
plt.plot(fdata)
plt.show()

# pandas가 지원하는 plot
fdata.plot()
fdata.plot(kind='bar')                    
fdata.plot(kind='box')                    
plt.xlabel("time")
plt.ylabel("data")
plt.show()

                    
                 



