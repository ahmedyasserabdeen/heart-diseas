from flask import Flask, request, jsonify
import numpy as np
import joblib

# Create a Flask app
app = Flask(__name__)

# Load the pre-trained model and preprocessor
model = joblib.load('voting.joblib')  # Save your trained model as rf_model.pkl
preprocessor = joblib.load('scaler.joblib')  # Save your preprocessor as preprocessor.pkl

@app.route('/')
def home():
    return "Heart Attack Prediction API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    # Get JSON data from the request
    input_data = request.json
    
    # Validate input
    required_features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
                         'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 
                         'ca', 'thal']
    for feature in required_features:
        if feature not in input_data:
            return jsonify({"error": f"Missing feature: {feature}"}), 400

    # Convert input data to a NumPy array for prediction
    data_array = np.array([input_data[feature] for feature in required_features]).reshape(1, -1)

    # Preprocess the input data
    preprocessed_data = preprocessor.transform(data_array)

    # Make a prediction
    prediction = model.predict(preprocessed_data)

    # Return the prediction
    result = {
        "prediction": int(prediction[0])}
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
