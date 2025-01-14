Here’s an updated and simpler README that highlights Flask usage:

---

# Heart Disease Prediction App & API

This project provides a **Flask**-based web application and API for predicting heart disease risk using a machine learning model.

---

## Features
- **Web Application:**
  - Predict heart disease risk from user inputs.
  - View interactive data visualizations.
- **API:**
  - POST endpoint for predictions via JSON input.

---

## Setup Instructions

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/your-repository/heart-disease-prediction.git
   cd heart-disease-prediction
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Web App:**
   ```bash
   python app.py
   ```

4. **Run the API:**
   ```bash
   python api.py
   ```

---

## API Usage

- **Endpoint:** `/predict`
- **Method:** `POST`
- **Input (JSON):**
  ```json
  {
      "age": 63, "sex": 1, "cp": 3, "trestbps": 145,
      "chol": 233, "fbs": 1, "restecg": 0, "thalach": 150,
      "exang": 0, "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1
  }
  ```
- **Response (JSON):**
  ```json
  {"prediction": 1}
  ```

---

## Files
- **`app.py`**: Web app logic.
- **`api.py`**: API for predictions.
- **`voting.joblib`**: Trained model file.
- **`scaler.joblib`**: Preprocessor file.
- **`heart.csv`**: Dataset file.

