import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

def train_best_model():
    data = pd.read_csv("data.csv")

    # Feature engineering
    data["distance_prep_interaction"] = (
        data["distance_km"] * data["prep_time_min"]
    )

    X = data[
        ["distance_km", "prep_time_min", "is_peak_hour", "distance_prep_interaction"]
    ]
    y = data["delivery_time_min"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(random_state=42)
    }

    best_model = None
    best_error = float("inf")

    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        error = mean_absolute_error(y_test, predictions)

        print(f"{name} MAE: {error:.2f}")

        if error < best_error:
            best_error = error
            best_model = model

    return best_model
