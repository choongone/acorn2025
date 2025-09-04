import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import wilcoxon

# [one-sample t 검정 : 문제1]  
# 영사기에 사용되는 구형 백열전구의 수명은 250시간이라고 알려졌다. 
# 한국연구소에서 수명이 50시간 더 긴 새로운 백열전구를 개발하였다고 발표하였다. 
# 연구소의 발표결과가 맞는지 새로 개발된 백열전구를 임의로 수집하여 수명시간 관련 자료를 얻었다. 
# 한국연구소의 발표가 맞는지 새로운 백열전구의 수명을 분석하라.
# 305 280 296 313 287 240 259 266 318 280 325 295 315 278

# 귀무 : 새로운 백열전구의 평균 수명은 300시간이다. 
# 대립 : 새로운 백열전구의 평균 수명은 300시간이 아니다. 

data = [305, 280, 296, 313, 287, 240, 259, 266, 318, 280, 325, 295, 315, 278]
print(np.array(data).mean())      # 289.7857142857143
result = stats.ttest_1samp(data, popmean=250)
print('statistic:%.5f, pvalue:%.5f'%result)     # statistic:6.06248, pvalue:0.00004
# pvalue < 0.05 
sns.displot(data, bins=10, kde=True, color='blue')
plt.xlabel('data')
plt.ylabel('value')
plt.show()
plt.close()

"""
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import wilcoxon

# 귀무 : 새로운 백열전구의 평균 수명은 300시간이다.
# 대립 : 새로운 백열전구의 평균 수명은 300시간이 아니다.
data = [305, 280, 296, 313, 287, 240, 259, 266, 318, 280, 325, 295, 315, 278]
fdata = pd.DataFrame(data, columns=['수명'])

print(fdata.mean())
print('정규성 검정:',stats.shapiro(fdata.수명)) # pvalue=0.8208613446833366 0.05보다 크므로 정규성 만족

# 정규성을 만족하면 T-검정 을사용 그렇지 않으면 윌콕슨 검정 사용


# wilcox_res = wilcoxon(fdata.수명 - 300) # 평균 300과 비교
# print('wilcox_res: ',wilcox_res) # pvalue=0.17679386538091857 0.05보다 크므로 귀무가설 채택


res = stats.ttest_1samp(fdata.수명, popmean=300) # pvalue:0.14361 0.05보다 크므로 귀무가설 채택
print('statistic:%.5f,pvalue:%.5f'%res)
# 결론 새로운 백열전구의 평균 수명은 300시간 이다.

sns.displot(fdata.수명, kde=True)
plt.show()
plt.close()
"""

# [one-sample t 검정 : 문제2] 
# 국내에서 생산된 대다수의 노트북 평균 사용 시간이 5.2 시간으로 파악되었다. 
# A회사에서 생산된 노트북 평균시간과 차이가 있는지를 검정하기 위해서 A회사 노트북 150대를 랜덤하게 선정하여 검정을 실시한다.
# 실습 파일 : one_sample.csv
# 참고 : time에 공백을 제거할 땐 ***.time.replace("     ", "")


df = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/one_sample.csv')
df = df.dropna(subset=['time'])      # 결측치 제거
df = df['time'].str.replace(r'\s+',"  ", regex=True)  # 앞뒤 공백 제거
df = df.str.split(expand=True).stack().reset_index(drop=True)
df = pd.to_numeric(df)
print(df.mean())
t_statistic, p_value = stats.ttest_1samp(a=df, popmean=5.2)
print(f'T-statistic: {t_statistic:.4f}')
print(f'P-value: {p_value:.4f}')

"""
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import wilcoxon

# 귀무 : A회사 노트북 평균 사용 시간이 5.2시간이다.
# 대립 : A회사 노트북 평균 사용 시간이 5.2시간이 아니다.
data = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/refs/heads/master/testdata_utf8/one_sample.csv')
data['time'] = pd.to_numeric(data['time'], errors='coerce') # 변환 할수 없는 데이터는 NaN 처리 여기서는 공백이 해당
df = data.dropna(axis=0) # NaN 행 삭제
print(df['time'].head(10))
print('정규성 검정:',stats.shapiro(df.time)) # pvalue=0.7242303336695732 0.05보다 크므로 정규성 만족

# 정규성을 만족하면 T-검정 을사용 그렇지 않으면 윌콕슨 검정 사용

# wilcox_res = wilcoxon(df.time - 5.2) # 평균 5.2과 비교
# print('wilcox_res: ',wilcox_res) # pvalue=0.00025724108542552436 0.05보다 작으므로 귀무가설 기각

res = stats.ttest_1samp(df.time, popmean=5.2) # pvalue:0.00014: 0.05보다 작으므로 귀무 가설 기각
print('statistic:%.5f,pvalue:%.5f'%res)

# 결론 A회사 노트북  평균 사용 시간은 5.2시간이 아니다.

sns.displot(df.time, kde=True)
plt.show()
plt.close()
"""

# [one-sample t 검정 : 문제3] 
# https://www.price.go.kr/tprice/portal/main/main.do 에서 
# 메뉴 중  가격동향 -> 개인서비스요금 -> 조회유형:지역별, 품목:미용 자료(엑셀)를 파일로 받아 미용 요금을 얻도록 하자. 
# 정부에서는 전국 평균 미용 요금이 15000원이라고 발표하였다. 이 발표가 맞는지 검정하시오.
# 월 단위 
# data2 = pd.read_csv('https://www.price.go.kr/tprice/portal/main/main.do')

# 귀무 : 전국 평균 미용 요금이 15000원이다.
# 대립 : 전국 평균 미용 요금이 15000원이 아니다.

data = pd.read_excel('testdata_11.csv')
df = data.drop(['번호', '품목', 'Unnamed: 2'],axis=1)
df = df.transpose()
df.rename(columns={0 :'금액'}, inplace=True)
df.dropna(inplace=True)
print('df.mean(): ', df.mean())  # 19512
print('정규성 검정:',stats.shapiro(df.금액)) # pvalue=0.05814403680264911 0.05보다 크므로 정규성 만족

# wilcox_res = wilcoxon(df.금액 - 15000) # 평균 15000과 비교
# print('wilcox_res: ',wilcox_res) # pvalue=3.0517578125e-05 0.05보다 작으므로 귀무가설 기각

# 정규성을 만족하면 T-검정을 사용, 그렇지 않으면 윌콕슨 검정 사용

res = stats.ttest_1samp(df.금액, popmean=5.15000) # pvalue:0.00000: 0.05보다 작으므로 귀무 가설 기각
print('statistic:%.5f,pvalue:%.5f'%res)

# 결론 전국 평균 미용 요금이 15000이 아니다.

sns.displot(df.금액, kde=True)
plt.show()
plt.close()



