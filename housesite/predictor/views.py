from django.shortcuts import render
from predictor.ml import predict_price

def predict_view(request, city):
    price = None

    if request.method == "POST":
        area = float(request.POST["area"])
        rooms = float(request.POST["rooms"])
        lat = float(request.POST["lat"])
        lon = float(request.POST["lon"])

        price = predict_price(area=area, rooms=rooms, lon=lon, lat=lat)

    return render(request, "predictor/predict.html", {"price": price})

def home(request):
    return render(request, "predictor/home.html")