from flask_cors import CORS
from flask import Flask
from Dataset_loc.describe import statistics_bp
from Dataset_loc.fetch_dataset import dataset_bp
from Dataset_loc.crop_prediction import motor_state_bp

app = Flask(__name__)

CORS(app, origins='*')

app.register_blueprint(statistics_bp)
app.register_blueprint(dataset_bp)
app.register_blueprint(motor_state_bp)

if __name__ == '__main__':
    app.run(debug=True)

