from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model
model = joblib.load("models/flood_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    values = [
        float(request.form["temp"]),
        float(request.form["humidity"]),
        float(request.form["cloud"]),
        float(request.form["annual"]),
        float(request.form["janfeb"]),
        float(request.form["marmay"]),
        float(request.form["junsep"]),
        float(request.form["octdec"]),
        float(request.form["avgjune"]),
        float(request.form["sub"])
    ]

    # Prediction
    prediction = model.predict([values])[0]

    # Confidence Score
    confidence = model.predict_proba([values])[0]
    confidence_percent = round(max(confidence) * 100, 2)

    if prediction == 1:
        result = "⚠️ High Flood Risk"
        recommendation = (
            "Stay alert. Avoid low-lying areas and follow official weather warnings."
        )
    else:
        result = "✅ Low Flood Risk"
        recommendation = (
            "No immediate flood risk. Continue monitoring weather updates."
        )

    return render_template(
        "index.html",
        prediction=result,
        confidence=confidence_percent,
        recommendation=recommendation,
        values=request.form
    )

if __name__ == "__main__":
    app.run(debug=True)