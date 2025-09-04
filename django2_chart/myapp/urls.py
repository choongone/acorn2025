from django.urls import path, include
from myapp import views

urlpatterns = [
    path('', views.show, name='show'),

]
