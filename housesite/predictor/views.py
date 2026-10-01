from django.shortcuts import render
from predictor.ml import predict_price

def predict_view(request, city):
    price = None
    form_data = {}
    if request.method == "POST":
        form_data = request.POST
        area = float(form_data["area"])
        rooms = float(form_data["rooms"])
        lat = float(form_data["lat"])
        lon = float(form_data["lon"])
        price = predict_price(area=area, rooms=rooms, lon=lon, lat=lat)

    return render(request, "predictor/predict.html", {"price": price, "form_data": form_data})

def home(request):
    return render(request, "predictor/home.html")