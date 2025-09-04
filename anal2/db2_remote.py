import MySQLdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rc('font', family='malgun gothic')
plt.rcParams['axes.unicode_minus'] = False
import sys
import pickle
import csv

"""conn = MySQLdb.connect(
    host = '127.0.0.1',
    user = 'root', 
    password = '1234',
    database ='mydb',
    port=3306,
    charset = 'utf8'
)
"""

"""config = {
    'host' : '127.0.0.1',
    'user' : 'root',
    'password' : '1234',
    'database' :'mydb',
    'port' : 3306,
    'charset' : 'utf8'
} """

try:
    with open('mymaria.dat',mode='rb')as obj:
        config = pickle.load(obj)

except Exception as e:
    print('읽기오류 : ', e)
    sys.exit()
try:
   
    conn = MySQLdb.connect(**config)   # 딕셔너리 unpacking
    cursor = conn.cursor() 
    sql = """
        select jikwonno, jikwonname ,busername, jikwonjik, jikwongen, jikwonpay from jikwon inner join buser
        on jikwon.busernum=buser.buserno
"""

    cursor.execute(sql)

    # 출력 1 : console
    # for (a,b,c,d,e,f) in cursor:
    for(jikwonno, jikwonname ,busername, jikwonjik, jikwongen, jikwonpay) in cursor:
        print(jikwonno, jikwonname ,busername, jikwonjik, jikwongen, jikwonpay)
    print()
    
    # 출력 2 : Dataframe
    df1 = pd.DataFrame(cursor.fetchall(),
                       columns=['jikwonno', 'jikwonname' ,'busername', 'jikwonjik', 'jikwongen', 'jikwonpay'])
    print(df1.head(3))
    print()

    # 출력 3 : csv 파일
    with open('jik_data.csv', mode='w', encoding='utf-8') as fobj:
        writer = csv.writer(fobj)
        for r in cursor:
            writer.writerow(r)
    # csv 파일을 읽어 dataFrame에 저장         
    df2 = pd.read_csv('jik_data.csv', header=None, 
                      names=['번호','이름','부서','직급','성별','연봉'])
    print(df2.head(3))

    print('\nDB의 자료를 pandas의 sql 처리 기능으로 읽기 -----')
    df = pd.read_sql(sql, conn)
    df.columns = ['번호','이름','부서','직급','성별','연봉']
    print(df.head(3))

    print('\nDB의 자료를 DataFrame으로 읽었으므로 pandas의 기능을 적용가능 ---')
    print('건수 : ', len(df))
    print('건수 : ', df['이름'].count())
    print('직급별 인원수 : ', df['직급'].value_counts())
    print('연봉 평균 : ', df.loc[:, '연봉'].mean())
    print()
    ctab = pd.crosstab(df['성별'], df['직급'], margins=True)  # 성별 직급별 건수 
    # print(ctab.to_html)

    # 시각화 - 직급별 연봉 평균 - pie 그래프
    jik_ypay = df.groupby(['직급'])['연봉'].mean()
    print('직급별 연봉 평균 : , jik_ypay')
    print(jik_ypay.index)
    print(jik_ypay.values)

    plt.pie(jik_ypay, explode=(0.2, 0, 0, 0.3, 0), labels=jik_ypay.index,
            shadow=True, 
            labeldistance=0.7,
            counterclock=False)
    plt.show()
    
except Exception as e:
    print('처리 오류: ', e)
finally:
    conn.close()


 