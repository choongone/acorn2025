from django.db import models

# database가 바뀐대도 장고의 source는 그대로 사용하겠다는 의미 
# aa.py에 있는 내용(Class)을 models.py에 붙여넣기 해야 orm을 사용가능

# Create your models here.
class Survey(models.Model):
    rnum = models.AutoField(primary_key=True)
    gender = models.CharField(max_length=4, blank=True, null=True)
    age = models.IntegerField(blank=True, null=True)
    co_survey = models.CharField(max_length=10, blank=True, null=True)

    class Meta:
        managed = False         # 테이블 생성 X, 기존 테이블 사용
        db_table = 'survey'     # MariaDB의 테이블명 # Class 이름은 바꿀 수 있지만 'survey'는 그대로 둬야 함.


 







