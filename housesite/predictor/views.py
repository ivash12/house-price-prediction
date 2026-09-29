from django.shortcuts import render

def predict_view(request, city):
    return render(request, "predictor/predict.html")

def home(request):
    return render(request, "predictor/home.html")