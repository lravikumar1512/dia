from flask import Flask, request, jsonify
import pandas as pd
import joblib

# Load trained artifacts
loaded_model = joblib.load("logistic_regression_model.joblib")
loaded_scaler = joblib.load("standard_scaler.joblib")
feature_names = joblib.load("feature_names.joblib")  # saved during training

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON input
        data = request.get_json(force=True)

        # Convert input to DataFrame with correct feature order
        input_df = pd.DataFrame([data])[feature_names]

        # Scale input
        scaled_data = loaded_scaler.transform(input_df)

        # Predict
        prediction = loaded_model.predict(scaled_data)

        return jsonify({
            "prediction": int(prediction[0])
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 400


if __name__ == "__main__":
    app.run(debug=True)

