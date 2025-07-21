# 💓 Heart Failure Prediction Web App

This is a machine learning web application that predicts the risk of heart failure based on clinical data. The app is built using **Flask**, and the model is trained using **XGBoost** on a public dataset.

---

## 📊 Dataset

- **Source**: [Heart Failure Clinical Records Dataset](https://www.kaggle.com/datasets/andrewmvd/heart-failure-clinical-data)
- **Features**: Age, anaemia, creatinine phosphokinase, diabetes, ejection fraction, high blood pressure, platelets, serum creatinine, serum sodium, sex, smoking, and follow-up time.
- **Target**: `DEATH_EVENT` (1 = death occurred, 0 = survived)

---

## 🧠 Model Training Summary

- Model used: `XGBoostClassifier`
- Preprocessing: `StandardScaler`
- Imbalance handling: `SMOTE`
- Evaluation: Accuracy, F1 Score, ROC AUC

---
## Flask Backend: app.py 

@app.route('/predict', methods=['POST'])
def predict():
    input_data = [float(x) for x in request.form.values()]
    input_scaled = scaler.transform([input_data])
    prediction = model.predict(input_scaled)

    result = "High Risk of Heart Failure" if prediction[0] == 1 else "Low Risk of Heart Failure"
    return render_template("index.html", prediction_text=result)
## index.html: Frontend
 <form action="/predict" method="post">
  <label>Age:</label>
  <input type="number" name="age" required>

  <label>Diabetes:</label>
  <select name="diabetes">
    <option value="1">Yes</option>
    <option value="0">No</option>
  </select>

  <!-- More fields here -->

  <button type="submit">Predict</button>
</form>

{% if prediction_text %}
  <h3>{{ prediction_text }}</h3>
{% endif %}
   

## To Run the App
use -> python app.py


