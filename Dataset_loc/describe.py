from flask import Blueprint, jsonify
from Dataset_loc.dataset import dataset

statistics_bp = Blueprint('statistics', __name__)

@statistics_bp.route('/api/statistics', methods=['GET'])
def get_statistics():
    data = dataset()
    summary = data.describe().to_dict()
    return jsonify(summary)
