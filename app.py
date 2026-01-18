from flask import Flask, request, jsonify
import pandas as pd

# Assuming loaded_model and loaded_scaler are already loaded from previous steps
# (e.g., loaded_model = joblib.load('logistic_regression_model.joblib'))
# (e.g., loaded_scaler = joblib.load('standard_scaler.joblib'))

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        # Convert input data to DataFrame, ensuring column order matches training features
        # X.columns represents the feature names from the original training data
        input_df = pd.DataFrame([data], columns=X.columns) # X is available from kernel state

        # Scale the input data
        scaled_data = loaded_scaler.transform(input_df)

        # Make prediction
        prediction = loaded_model.predict(scaled_data)

        # Return prediction as JSON response
        return jsonify({'prediction': int(prediction[0])})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

# To run the Flask application locally
# This block will only execute if the script is run directly (not imported)
if __name__ == '__main__':
    # For deployment in Colab, ngrok is typically used to expose the local server
    # For local testing, you can run app.run(debug=True)
    print("Flask app is ready. To run, use app.run(debug=True) or set up ngrok for external access.")
