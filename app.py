from flask import Flask, render_template, request
import joblib
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = joblib.load('model/model.pkl')

# Home route
@app.route('/')
def home():
    return render_template('crop.html')

# Area conversion to hectares
def convert_area_to_hectares(value, unit):
    conversions = {
        "hectares": 1,
        "acres": 0.4047,
        "squareKilometers": 100,
        "squareMeters": 0.0001,
        "squareFeet": 0.0000092903,
        "cents": 0.00404686,
        "bighas": 0.2529,        # varies regionally
        "roods": 0.1012,
        "perches": 0.002529,
        "poles": 0.002529,
        "yards": 0.00008361,
        "miles": 258.999,
        "kilometers": 100,       # assuming square
        "rods": 0.002529,
        "chains": 0.4047
    }
    return value * conversions.get(unit, 1)

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Required inputs
        rainfall = float(request.form['rainfall'].strip())
        temperature = float(request.form['temperature'].strip())
        soil_ph = float(request.form['soil_ph'].strip())

        # Optional extended inputs (not used by model yet)
        area = float(request.form['area'].strip())
        area_unit = request.form['areaUnit']
        crop = request.form['crop']
        soil = request.form['soil']
        season = request.form['season']

        # Convert area to hectares
        area_hectares = convert_area_to_hectares(area, area_unit)

        # Load input template and set values
        input_df = pd.read_csv('model/feature_template.csv')
        input_df.loc[0] = 0  # Reset the first row to zeros

        # Update only the features used by model
        input_df.loc[0, 'average_rain_fall_mm_per_year'] = rainfall
        input_df.loc[0, 'avg_temp'] = temperature
        input_df.loc[0, 'pesticides_tonnes'] = soil_ph
        input_df.loc[0, 'Year'] = 2023
        input_df.loc[0, 'Area'] = 'India'
        input_df.loc[0, 'Item'] = 'Rice'  # Placeholder - adjust if your model supports dynamic crop types

        print("✅ Form inputs:")
        print(f"Rainfall: {rainfall}, Temp: {temperature}, pH: {soil_ph}, Area: {area} ({area_unit} = {area_hectares} hectares)")
        print("🚀 Input DataFrame:\n", input_df.head())

        # Make prediction
        prediction = model.predict(input_df)[0]

        return render_template('crop.html', prediction=round(prediction, 2))

    except Exception as e:
        print("🔥 ERROR:", e)
        return render_template('crop.html', error="⚠️ Something went wrong while predicting.")

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
