from django.urls import path

from . import views

urlpatterns = [
    path("", views.home, name="home"),
    path("about/", views.about, name="about"),
    path("model/", views.model, name="model"),
    # path("hello/", views.hello_world, name="hello_world"),
]
