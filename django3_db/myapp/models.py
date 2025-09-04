from django.db import models

# Create your models here.
"""
class Jikwon(models.Model):
    jikwonno = models.IntegerField(primary_key=True)
    jikwonname = models.CharField(max_length=10)
    busernum = models.IntegerField()
    jikwonjik = models.CharField(max_length=10, blank=True, null=True)
    jikwonpay = models.IntegerField(blank=True, null=True)
    jikwonibsail = models.DateField(blank=True, null=True)
    jikwongen = models.CharField(max_length=4, blank=True, null=True)
    jikwonrating = models.CharField(max_length=3, blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'jikwon'
        """
class Buser(models.Model):
    buser_no = models.AutoField(primary_key=True)
    buser_name = models.CharField(max_length=50)
    buser_loc = models.CharField(max_length=50)

    def __str__(self):
        return self.buser_name

class Jikwon(models.Model):
    jikwon_no = models.AutoField(primary_key=True)
    jikwon_name = models.CharField(max_length=50)
    jikwon_jik = models.CharField(max_length=50)  # 직급
    jikwon_gen = models.CharField(max_length=2)   # 성별
    jikwon_pay = models.IntegerField()
    jikwon_ibsail = models.DateField()
    buser = models.ForeignKey(Buser, on_delete=models.CASCADE)

    def __str__(self):
        return self.jikwon_name