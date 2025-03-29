from flask import Flask, jsonify, request
from flask_cors import CORS  # To allow frontend requests

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend access

@app.route('/start', methods=['GET'])
def start_workout():
    return jsonify({'message': 'Workout Started!'})

if __name__ == '__main__':
    app.run(debug=True)
