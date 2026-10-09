from django.http import Http404
from django.shortcuts import render
from predictor.ml import CITIES, predict_price, is_inside_city

CURRENCY = {"amsterdam": "€", "new-york": "$"}


def predict_view(request, city):
    if city not in CITIES:
        raise Http404("Unknown city")

    fields = [f for f in CITIES[city]["features"] if f not in ("lat", "lon", "distance")]

    price = None
    error = None
    form_data = {}

    if request.method == "POST":
        form_data = request.POST
        try:
            features = {name: float(form_data[name]) for name in fields + ["lat", "lon"]}
            if not is_inside_city(city, features["lat"], features["lon"]):
                error = "This address is outside the supported area."
            else:
                price = predict_price(city, features)
        except (ValueError, KeyError):
            error = "Please fill in all fields with valid numbers."

    return render(request, "predictor/predict.html", {
        "city": city,
        "currency": CURRENCY[city],
        "fields": fields,
        "price": price,
        "error": error,
        "form_data": form_data,
    })


def home(request):
    return render(request, "predictor/home.html")