from django.shortcuts import render
from django.conf import settings
from pathlib import Path
import seaborn as sns
import matplotlib
matplotlib.use('Agg')  # matplotlib이 그래프를 그릴 때 backend로 지정하는 코드 
# 기본은 그래프 창이 열리는데, Agg,PDF -> GUI 없이 시각화를 파일로 저장 
import matplotlib.pyplot as plt


# Create your views here.
def main(request):
    return render(request, 'main.html')

def showdata(request):
    df = sns.load_dataset('iris')
    # image 저장 경로를 설정  <BASE_DIR/STATIC/images/iris.png
    static_app_dir = Path(settings.BASE_DIR) / 'static' / 'images'
    static_app_dir.mkdir(parents=True, exist_ok=True)
    img_path = static_app_dir/ 'iris.png'
    
    # 파이 파트 저장
    counts = df['species'].value_counts().sort_index()
    print('counts : ', counts)
    plt.figure()
    counts.plot.pie(autopct='%1.1f%%', startangle=90, ylabel='')  # 소수 첫째자리까지 출력 
    plt.title('iris species count')
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(img_path, dpi=130)
    plt.close()

    # df를 table tag 만들어 show.html에 전달
    table_html = df.to_html(classes='table-striped table-sm', index=False)

    return render(request, 'show.html', {
            'table':table_html,
            'img_relpath': 'images/iris.png'

    })
