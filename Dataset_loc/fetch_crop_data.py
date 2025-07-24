import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from flask import Blueprint, jsonify, request
import requests
from Dataset_loc.dataset import dataset
from Dataset_loc.laravel_url_api import Laravel_url_api

motor_state_bp = Blueprint('motor_state_bp', __name__)

@motor_state_bp.route('/api/state', methods=['GET'])
def get_motor_state():
    
    # Fetch data from Laravel API
    response = requests.get(Laravel_url_api)
    
    if response.status_code != 200:
        return jsonify({"message": "Failed to fetch data from Laravel API"}), 400

    data_from_laravel = response.json()

    required_fields = ['temperature', 'vibration']
    
    if not all(field in data_from_laravel for field in required_fields):
        return jsonify({"message": "Missing necessary data for prediction"}), 400

    # Load dataset
    data = dataset()  

    X = data[['temperature', 'vibration']]
    y = data['state']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)

    # Prepare new data for prediction
    new_data = pd.DataFrame({
        "temperature": [data_from_laravel['temperature']],
        "vibration": [data_from_laravel['vibration']]
    })

    # Validate input values
    if new_data['temperature'][0] <= 0 or new_data['vibration'][0] < 0:
        return jsonify({"message": "Invalid input values for prediction."}), 400

    # Make prediction
    predicted_probabilities = model.predict_proba(new_data)

    # Convert to DataFrame for readability
    predictions_df = pd.DataFrame(predicted_probabilities, columns=model.classes_)
    filtered_predictions = predictions_df.T[predictions_df.T[0] > 0].sort_values(by=0, ascending=False)

    if filtered_predictions.empty:
        return jsonify({"message": "Unable to determine motor state."})
    
    else:
        most_probable_state = filtered_predictions.index[0]
        return jsonify({
            "predicted_probabilities": filtered_predictions.to_dict(),
            "most_probable_state": most_probable_state
        })