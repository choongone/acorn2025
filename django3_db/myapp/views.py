from django.shortcuts import render
from django.db import connection
from django.utils.html import escape
import pandas as pd
from datetime import datetime
import json


# Create your views here.
def indexFunc(request):
    dept = request.GET.get("dept", "").strip()
    sql = """
        select b.buserno as 부서번호,
        j.jikwonname as 직원명,
        b.busername as 부서명, 
        j.jikwonjik as 직급,
        j.jikwonpay as 연봉, 
        j.jikwonibsail as 입사일,
        j.jikwongen as 성별
        from jikwon j inner join buser b
        on j.busernum = b.buserno
    """

    params = []
    if dept:
        sql += " WHERE b.busername LIKE %s" # 부서 필터링
        params.append(f"%{dept}%")
    
    # 부서번호, 직원명 순으로 오름차순 정렬
    sql += " ORDER BY b.buserno ASC, j.jikwonname ASC"

    with connection.cursor() as cursor:
        cursor.execute(sql, params)
        rows = cursor.fetchall()
        cols = [col[0] for col in cursor.description]
    df = pd.DataFrame(rows, columns=cols)
    # 근무년수 계산
    if not df.empty:
        current_year = datetime.now().year
        df['근무년수'] = df['입사일'].apply(lambda x: 
            current_year - x.year if pd.notnull(x) else 0
        )
    # print('DataFrame:', df.head(3))

    # DataFrame를 HTML 테이블로 변환
    # 사번, 직원명, 부서명, 직급, 연봉, 근무년수
    if not df.empty:
        basic_df = df[['부서번호', '직원명', '부서명', '직급', '연봉', '근무년수']].copy()
        
         # 숫자 컬럼에 CSS 스타일 적용
        formatters = {
            '부서번호': lambda x: f'<div style="text-align: center">{x}</div>',
            '직원명': lambda x: f'<div style="text-align: center">{x}</div>',
            '부서명': lambda x: f'<div style="text-align: center">{x}</div>',
            '직급': lambda x: f'<div style="text-align: center">{x}</div>',
            '연봉': lambda x: f'<div style="text-align: right">{x:,}</div>',  # 천단위 구분자도 추가
            '근무년수': lambda x: f'<div style="text-align: right">{x}</div>'
            }
    
        basic_html = basic_df.to_html(
            index=False, 
            escape=False, 
            justify='center',
            formatters=formatters, 
            classes='table table-sm table-striped table-bordered')
    else:
        basic_html = "<p>데이터가 없습니다</p>"

    # 부서별, 직급별 연봉합, 연봉평균
    if not df.empty:
        buser_salary_df = df.groupby('부서명')['연봉'].agg(['sum', 'mean']).reset_index()
        buser_salary_df.columns = ['부서명', '연봉합계', '연봉평균']
        # 포맷터 정의
        buser_formatters = {
            '부서명': lambda x: f'<div style="text-align: center">{x}</div>',
            '연봉합계': lambda x: f'<div style="text-align: right">{x:,}</div>',
            '연봉평균': lambda x: f'<div style="text-align: right">{x:,}</div>'
        }
        
        buser_salary_html = buser_salary_df.to_html(
            index=False, 
            escape=False, 
            formatters=buser_formatters,
            justify='center', 
            classes='table table-sm table-striped table-bordered')
        # print('buser_salary_df:', buser_salary_df)

        jikjik_salary_df = df.groupby('직급')['연봉'].agg(['sum', 'mean']).reset_index()
        jikjik_salary_df.columns = ['직급', '연봉합계', '연봉평균']
        # 포맷터
        jikjik_formatters = {
            '직급': lambda x: f'<div style="text-align: center">{x}</div>',
            '연봉합계': lambda x: f'<div style="text-align: right">{x:,}</div>',
            '연봉평균': lambda x: f'<div style="text-align: right">{x:,}</div>'
        }

        jikjik_salary_html = jikjik_salary_df.to_html(
            index=False, 
            escape=False, 
            formatters=jikjik_formatters,
            justify='center', 
            classes='table table-sm table-striped table-bordered')

    else:
        buser_salary_html = "<p>데이터가 없습니다</p>"
        jikjik_salary_html = "<p>데이터가 없습니다</p>"

    # 부서별 연봉합, 평균으로 그래프용 json파일
    buser_labels = buser_salary_df['부서명'].tolist()
    buser_means = buser_salary_df['연봉평균'].tolist()
    buser_sums = buser_salary_df['연봉합계'].tolist()
    print('buser_labels:', buser_labels)
    print('buser_means:', buser_means)
    print('buser_sums:', buser_sums)

    # 성별, 직급별 빈도표
    # 성별, 직급별 빈도표
    if not df.empty:
        # crosstab 방식 (추천)
        gender_position_df = pd.crosstab(
            df['성별'], 
            df['직급'], 
        )
        
        gender_position_html = gender_position_df.to_html(
            index=True, 
            justify='center', 
            classes='table table-striped table-bordered'
        )
    else:
        gender_position_html = "<p>성별, 직급별 데이터가 없습니다.</p>"
    ctx_dict = {
        'dept': dept,
        'basic_html': basic_html,
        'buser_salary_html': buser_salary_html,
        'jikjik_salary_html': jikjik_salary_html,
        'buser_labels_json': json.dumps(buser_labels),
        'buser_means_json': json.dumps(buser_means),
        'buser_sums_json': json.dumps(buser_sums),
        'gender_position_html': gender_position_html
    }
    return render(request, "index.html", ctx_dict)
