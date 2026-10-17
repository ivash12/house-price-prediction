# House Price Estimator

A web app that estimates property prices in **Amsterdam**, **New York City** and **Paris**
using neural networks built with TensorFlow. Pick a city, enter a few details about the
property, start typing its address, and get an estimated price.

## Features

- One TensorFlow model per city, trained on public housing data
- Django web interface with a city picker and an About page
- Address autocomplete limited to the chosen city ([Photon](https://photon.komoot.io/), OpenStreetMap data)
- Engineered feature: distance from the city centre

## How it works

1. **Data cleaning** (pandas): remove duplicates, missing values, outliers and
   placeholder values. For Paris, only single-apartment sales are kept, because
   multi-unit sales share one total price.
2. **Features**: size, rooms (or beds/baths for NYC), latitude, longitude and distance
   from the city centre.
3. **Model**: Keras `Sequential` network: `Normalization` layer, two `Dense(64, relu)`
   layers and one output. It is trained on `log(price)` with MAE loss, and predictions are
   converted back with `exp()`.
4. **Web app**: Django loads all models once at startup. On submit, the view checks that
   the address is inside the city and passes the features to the model.

## Results

| City | Data | Median error |
|---|---|---|
| Amsterdam | ~900 listings (2021, asking prices) | ~13% |
| New York City | a few thousand listings | ~20% |
| Paris | ~30,000 apartment sales (2025, official records) | ~13% |

These are rough estimates for learning purposes, not property valuations. The models only
know size, rooms and location, not condition, floor, view or renovation.

## Project structure

```
house-price-prediction/
├── model_training.py          # Amsterdam model
├── model_training_nyc.py      # New York model
├── model_training_paris.py    # Paris model
├── requirements.txt
└── housesite/                 # Django project
    ├── manage.py
    ├── housesite/             # settings, root URLs
    └── predictor/             # the app
        ├── ml.py              # loads models, runs predictions
        ├── views.py
        ├── urls.py
        ├── *.keras            # trained models
        ├── templates/predictor/        # home, predict, about pages
        └── static/predictor/script.js  # address autocomplete
```

## Running locally

Requires Python 3.14 (the version this project was built with).

```bash
git clone https://github.com/<your-username>/house-price-prediction.git
cd house-price-prediction
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Django needs a secret key to run. It is kept out of the repository, so you need to create your own:

1. Generate a key (with the virtual environment active):

   ```bash
   python -c "from django.core.management.utils import get_random_secret_key; print(get_random_secret_key())"
   ```

2. Create a file named `.env` in `housesite/housesite/` (the folder that contains `settings.py`)
   and paste the key into it:

   ```
   SECRET_KEY=paste-the-generated-key-here
   ```

   Don't add quotes or spaces around `=`. The `.env` file is listed in `.gitignore`, so it
   is never committed.

Then start the server:

```bash
cd housesite
python manage.py runserver
```

Open http://127.0.0.1:8000/

## Data sources

- Amsterdam: [Housing Prices in Amsterdam (Kaggle)](https://www.kaggle.com/datasets/thomasnibb/amsterdam-house-price-prediction)
- New York: [New York Housing Market (Kaggle)](https://www.kaggle.com/datasets/nelgiriyewithana/new-york-housing-market)
- Paris: [DVF géolocalisées (data.gouv.fr)](https://www.data.gouv.fr/datasets/demandes-de-valeurs-foncieres-geolocalisees/), Paris (département 75), 2025

## Tech stack

Python · TensorFlow/Keras · pandas · NumPy · scikit-learn · Django · JavaScript
