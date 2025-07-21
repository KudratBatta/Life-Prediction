from flask import Flask, request, render_template
import numpy as np
import joblib

app = Flask(__name__)

# Load model and scaler
model = joblib.load("xgb_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(request.form[key]) for key in request.form]
    scaled_features = scaler.transform([features])
    prediction = model.predict(scaled_features)[0]
    return render_template("index.html", prediction_text=f"Predicted DEATH_EVENT: {prediction}")

if __name__ == "__main__":
    app.run(debug=True)
