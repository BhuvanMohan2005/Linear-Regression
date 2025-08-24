from django.urls import path
from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("add-point/", views.add_point, name="add_point"),
    path("reset/", views.reset_points, name="reset_points"),
]

