from django.urls import path
from . import views

urlpatterns = [
    path("", views.home),
    path("<str:city>/", views.predict_view),
]
