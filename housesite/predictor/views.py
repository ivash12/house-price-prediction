from django.http import HttpResponse
from django.shortcuts import render

# Create your views here.
def predict_view(request, city):
    return render(request, "predictor/predict.html")