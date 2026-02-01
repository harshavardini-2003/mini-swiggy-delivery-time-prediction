from src.train_model import train_best_model
from src.predict import predict_delivery_time

if __name__ == "__main__":
    model = train_best_model()

    # Sample order details
    distance_km = 5
    prep_time_min = 15
    is_peak_hour = 1

    predicted_time = predict_delivery_time(
        model,
        distance_km,
        prep_time_min,
        is_peak_hour
    )

    print(f"\nPredicted delivery time: {predicted_time} minutes")
