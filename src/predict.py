def predict_delivery_time(model, distance_km, prep_time_min, is_peak_hour):
    interaction = distance_km * prep_time_min

    prediction = model.predict([[
        distance_km,
        prep_time_min,
        is_peak_hour,
        interaction
    ]])

    return round(prediction[0], 2)
