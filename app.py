from flask import Flask, render_template, request
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load model
model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("crop.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        try:
            # Collect features from form input
            features = [float(x) for x in request.form.values()]
            final_features = np.array([features])
            prediction = model.predict(final_features)[0]

            return render_template("crop.html", prediction_text=f"Predicted Crop Yield: {prediction}")
        except Exception as e:
            return render_template("crop.html", prediction_text=f"Error: {e}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Render uses this
    app.run(host="0.0.0.0", port=port, debug=False)
