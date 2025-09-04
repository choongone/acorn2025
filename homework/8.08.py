# a) MariaDB에 저장된 jikwon, buser, gogek 테이블을 이용하여 아래의 문제에 답하시오.
#      - 사번 이름 부서명 연봉, 직급을 읽어 DataFrame을 작성
#      - DataFrame의 자료를 파일로 저장
#      - 부서명별 연봉의 합, 연봉의 최대/최소값을 출력
#      - 부서명, 직급으로 교차 테이블(빈도표)을 작성(crosstab(부서, 직급))
#      - 직원별 담당 고객자료(고객번호, 고객명, 고객전화)를 출력. 담당 고객이 없으면 "담당 고객  X"으로 표시
#      - 부서명별 연봉의 평균으로 가로 막대 그래프를 작성

import MySQLdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rc('font', family='malgun gothic')      #폰트를 반드시 지정해줘야 함. 한글이 깨질 수 있음.
plt.rcParams['axes.unicode_minus'] = False  #한글 글꼴을 사용할 시 음수가 깨질 수 있기 때문에 해야함.
import sys
import pickle
import csv

try:
    with open('myMaria.dat', mode='rb') as obj:
        config = pickle.load(obj)                   #피클스를 이용해 다른파일(myMaria.dat)에 미리 저장한 접속정보 부르기
except Exception as error:
    print('readError! : fail to read myMara', error)#접속정보 부르기에 실패했을 경우의 예외처리
    sys.exit()

try:
    #A1 : 사번 이름 부서명 연봉, 직급을 읽어 DataFrame을 작성
    connect = MySQLdb.connect(**config)     #피클스로 불러온 myMaria.dat의 내용물 언패킹(**)
    cursor = connect.cursor()
                                            #SQL 문법을 그대로 활용해서 jikwon 테이블에 buser 테이블은 이너조인해서 읽어오기
    sql = """
        SELECT jikwonno, jikwonname, busername, jikwonpay, jikwonjik
        FROM jikwon INNER JOIN buser
        ON jikwon.busernum = buser.buserno
    """
    cursor.execute(sql)                     #위의 SQL문 실행, cursor에 내용물 임시저장
    myDf = pd.DataFrame(cursor.fetchall(), columns=['사번', '이름', '부서명', '연봉', '직급'])  #SQL문으로 읽어온 내용을 모두 활용해 데이터프레임 만들기, 칼럼명 임의지정
    
    
    #A2 : DataFrame의 자료를 파일로 저장
    with open('IHateNamingFiles.csv', mode = 'w', encoding='utf-8') as obj: #쓰기모드(w), utf8 인코딩(한글문제)으로 새로운 CSV 파일 읽어오기
        writer = csv.writer(obj)
        for r in cursor:
            writer.writerow(r)
    
    
    #A3 : 부서명별 연봉의 합, 연봉의 최대/최소값을 출력
    print('부서명 별 연봉의 합 : ')
    filteredDf = myDf[myDf['부서명'] == '총무부']   #기존 데이터프레임에서 부서명이 총무부인 것들만 읽어온 새 데이터프레임 만들기
    print('총무부 연봉의 합 : ', filteredDf['연봉'].sum(), '최소값 : ', filteredDf['연봉'].min(), '최대값 : ', filteredDf['연봉'].max())    #해당 데이터프레임에서 사용할 정보들만 읽어오기
    filteredDf = myDf[myDf['부서명'] == '영업부']
    print('영업부 연봉의 합 : ', filteredDf['연봉'].sum(), '최소값 : ', filteredDf['연봉'].min(), '최대값 : ', filteredDf['연봉'].max())
    filteredDf = myDf[myDf['부서명'] == '전산부']
    print('전산부 연봉의 합 : ', filteredDf['연봉'].sum(), '최소값 : ', filteredDf['연봉'].min(), '최대값 : ', filteredDf['연봉'].max())
    filteredDf = myDf[myDf['부서명'] == '관리부']
    print('관리부 연봉의 합 : ', filteredDf['연봉'].sum(), '최소값 : ', filteredDf['연봉'].min(), '최대값 : ', filteredDf['연봉'].max())
    
    
    #A4 : 부서명, 직급으로 교차 테이블(빈도표)을 작성(crosstab(부서, 직급))
    ctab = pd.crosstab(myDf['부서명'], myDf['직급'])        #ctab에 두 데이터프레임을 크로스테이블한 새 데이터프레임 만들기
    print('부서명, 직급으로 교차 테이블')
    print(ctab)                                            #해당 데이터프레임 그대로 출력
    
    
    #A5 : 직원별 담당 고객자료(고객번호, 고객명, 고객전화)를 출력. 담당 고객이 없으면 "담당 고객  X"으로 표시
    sql2 = """
        SELECT gogekno, gogekname, gogektel, gogekdamsano
        FROM gogek
    """
    cursor.execute(sql2)    #두 번째 SQL문법을 활용해 gogek 테이블에서 고객번호, 고객명, 고객전화번호, 고객담당자 사원번호 읽어와서 커서에 저장
    myDf2 = pd.DataFrame(cursor.fetchall(), columns=['고객번호', '고객명', '고객전화', '사번']) #위에서 읽어온 SQL문으로 새 데이터프레임 만들기
    myDf3 = pd.merge(myDf, myDf2, on='사번', how='outer')             #두개의 데이터프레임을 아우터조인으로 합치기
    myDf3['고객명'].fillna('담당 고객 X', inplace=True)                 #NaN으로 뜨는 값들 중 '고객명'에 한정해서 '담당 고객 X'로 뜨게하기
    print(myDf3[['이름', '고객번호', '고객명', '고객전화']])
    
    
    #A6 : 부서명별 연봉의 평균으로 가로 막대 그래프를 작성
    moneyData = myDf.groupby(['부서명'])['연봉'].mean()     #matplot 제작에 활용할 데이터를 미리 저장
    #print(moneyData)
    plt.figure(figsize=(8, 6))                              #그래프 창 크기
    plt.barh(moneyData.index, moneyData.values)             #가로 막대그래프, x축 데이터, y축 데이터
    plt.title('부서명별 연봉의 평균')                          #그래프 제목 지정
    plt.show()                                              #그래프 보이기

except Exception as error:
    print('failed to load MariaDB!', error)
finally:
    cursor.close()
    connect.close()         ##데이터베이스 작업이 끝나면 반드시 실행해야함.

#  b) MariaDB에 저장된 jikwon 테이블을 이용하여 아래의 문제에 답하시오.
#      - pivot_table을 사용하여 성별 연봉의 평균을 출력
#      - 성별(남, 여) 연봉의 평균으로 시각화 - 세로 막대 그래프
#      - 부서명, 성별로 교차 테이블을 작성 (crosstab(부서, 성별))

import MySQLdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import pickle
import csv
from pandas import DataFrame, Series

# pandas 문제 7)
# b) MariaDB에 저장된 jikwon 테이블을 이용하여 아래의 문제에 답하시오.
#      - pivot_table을 사용하여 성별 연봉의 평균을 출력
#      - 성별(남, 여) 연봉의 평균으로 시각화 - 세로 막대 그래프
#      - 부서명, 성별로 교차 테이블을 작성 (crosstab(부서, 성별))


plt.rc('font', family= 'malgun gothic')     # 한글 깨짐 방지 코드 두줄
plt.rcParams['axes.unicode_minus']= False   # 한글 깨짐 방지 코드 두줄

try:
    with open('./mymaria.dat', mode = 'rb') as obj:
        config = pickle.load(obj)

except Exception as e:
    print('읽기 오류 : ',e)
    sys.exit()

try:
    conn = MySQLdb.connect(**config)
    cursor = conn.cursor()
    
    sql = '''
        select jikwonname,busername, jikwonjik, jikwongen, jikwonpay
        from jikwon inner join buser
        on jikwon.busernum = buser.buserno
            '''
    cursor.execute(sql)
    df2 = pd.DataFrame(cursor.fetchall(),
                      columns = ['jikwonname','busername',
                                 'jikwonjik','jikwongen','jikwonypay']
                      )

#      - pivot_table을 사용하여 성별 연봉의 평균을 출력
    # pivot_table(테이블을 생성할 원본, index = 행으로 사용할 열들의 이름, cloumns = 열로 사용할 열들의 이름, values = 집계 대상 열 이름)
    df2.pivot_table(index = 'jikwongen', values = 'jikwonypay')

    # 실행할때 주석 해제
    # print(df2)

    man = df2[df2['jikwongen'] == '남']
    woman = df2[df2['jikwongen'] == '여']

    mean_manYpay = round(man['jikwonypay'].mean(),2)
    mean_womanYpay = round(woman['jikwonypay'].mean(),2)
    print('남성 직원 평균 연봉 : ', mean_manYpay)
    print('여성 직원 평균 연봉 : ', mean_womanYpay)
    
#      - 성별(남, 여) 연봉의 평균으로 시각화 - 세로 막대 그래프

    # bar(x좌표 내용, y좌표 내용 - 변수 또는 문자열 사용 가능) 클래스 사용
    labels = ['남성 직원 평균 연봉', '여성 직원 평균 연봉']
    values = [mean_manYpay,mean_womanYpay]
    plt.bar(labels, values)
    plt.title(config['database'])
    plt.ylabel('단위 : (만) 원')

    # 실행할때 주석 해제
    plt.show()

#      - 부서명, 성별로 교차 테이블을 작성 (crosstab(부서, 성별))
    # 변수의 빈도를 계산해줌
    # crosstab(index - 행으로 쓸 변수, columns - 열로 쓸 변수)
    # 각각의 busername에 jikwongen의 요소값(성별)이 몇번 나오는지 출력해줌
    ctab2 = pd.crosstab(df2['busername'],df2['jikwongen'])
    print(ctab2)
    print(df2.head(3))

except Exception as e:
    print('처리 오류 : ', e)
finally:
    conn.close()

#  c) 키보드로 사번, 직원명을 입력받아 로그인에 성공하면 console에 아래와 같이 출력하시오.
#       조건 :  try ~ except MySQLdb.OperationalError as e:      사용
#      사번  직원명  부서명   직급  부서전화  성별...
#      인원수 : * 명

import MySQLdb
import pandas as pd

conn = MySQLdb.connect(
    host='127.0.0.1',
    user='root',
    password='1234',
    database='mydb',
    port=3306,
    charset='utf8'
)

try:    
    cursor = conn.cursor()
    no = int(input('사원번호 입력:')) # 사원번호 입력
    name = input('이름 입력:') # 이름 입력

    # DB에 입력받은 no, name로 검색
    sql_login = """
        select jikwonno, jikwonname
        from jikwon
        where jikwonname = %s and jikwonno = %s 
    """
    cursor.execute(sql_login, (name,no)) # 명령어 실행/ 입력 받은 name,no 맵핑
    result = cursor.fetchone() # 일치한 행 하나 가져오기
    
    if result: # 일치한 값이 있을 때 실행
        print('로그인 성공')
        
        # 전체 조회할 수 있는 sql 명령어
        sql_all="""
            select jikwonno, jikwonname, buser.busername, jikwonjik, buser.busertel, jikwongen
            from jikwon inner join buser
            on jikwon.busernum=buser.buserno
        """
        cursor.execute(sql_all) # 명령어 실행
        # DataFrame에 담기
        df = pd.DataFrame(cursor.fetchall(), columns=[
            '사번', '직원명','부서명','직급', '부서전화','성별'
        ])
        print(df.head(3))
        print(f'인원수 :{df["사번"].count()}명') # 직원 전체 인원수 출력
        
    else:
        print('로그인 실패')

except MySQLdb.OperationalError as e: # DB접속, 인증, 네트워크 문제
    print(f'문제(2) 발생: {e}')

except Exception as e:
    print(f'문제 발생 :{e}')

finally:
    cursor.close()
    conn.close()