import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from flask import Blueprint, jsonify, request
import requests
from Dataset_loc.dataset import dataset  # Load local dataset
from Dataset_loc.laravel_url_api import Laravel_url_api  # Laravel API URL

motor_state_bp = Blueprint('motor_state_bp', __name__)

# Load dataset and train model once
data = dataset()

X = data[['temperature', 'vibration']]
y = data['state']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=0)
model.fit(X_train, y_train)

# Compute model accuracy
model_accuracy = round(model.score(X_test, y_test) * 100, 2)  # Accuracy in percentage


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

    # Convert string values to float
    try:
        temp = float(data_from_laravel['temperature'])
        vib = float(data_from_laravel['vibration'])
    except ValueError:
        return jsonify({"message": "Invalid data format. Temperature and Vibration should be numeric."}), 400

    # Validate input values
    if temp <= 0 or vib < 0:
        return jsonify({"message": "Invalid input values for prediction."}), 400

    # Prepare new data for prediction
    new_data = pd.DataFrame({"temperature": [temp], "vibration": [vib]})

    # Make prediction
    predicted_probabilities = model.predict_proba(new_data)

    # Convert to DataFrame for readability
    predictions_df = pd.DataFrame(predicted_probabilities, columns=model.classes_)
    filtered_predictions = predictions_df.T[predictions_df.T[0] > 0].sort_values(by=0, ascending=False)

    if filtered_predictions.empty:
        return jsonify({"message": "Unable to determine motor state."})
    else:
        most_probable_state = filtered_predictions.index[0]
        most_probable_prob = filtered_predictions.iloc[0, 0]

        return jsonify({
            "model_accuracy": model_accuracy,  # Precomputed accuracy
            "most_probable_state": most_probable_state,
            "predicted_probability": round(most_probable_prob, 2)  # Show only top prediction
        })
