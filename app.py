import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, render_template

app = Flask(__name__)

# Load trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/")
def home():
    return render_template("crop.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Collect form data
        features = [float(x) for x in request.form.values()]
        final_features = np.array(features).reshape(1, -1)

        # Predict crop
        prediction = model.predict(final_features)[0]

        return render_template("crop.html", prediction_text=f"Recommended Crop: {prediction}")

    except Exception as e:
        return render_template("crop.html", prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
