from flask import Flask, request, jsonify
import joblib
import numpy as np


app = Flask(__name__)


# Load trained model
model = joblib.load(
    "models/predictive_maintenance_model.pkl"
)


@app.route("/")
def home():

    return "AI Predictive Maintenance API is running"


@app.route("/predict", methods=["POST"])
def predict():

    try:

        data = request.get_json()

        if data is None:
            return jsonify({
                "error": "No JSON data received"
            }), 400


        # Read sensor values
        temperature = float(
            data.get("temperature", 0)
        )

        vibration = float(
            data.get("vibration", 0)
        )

        current = float(
            data.get("current", 0)
        )

        rpm = float(
            data.get("rpm", 0)
        )


        # Prepare data for model
        X = np.array([
            [
                temperature,
                vibration,
                current,
                rpm
            ]
        ])


        # Prediction
        prediction = model.predict(X)[0]


        # Health score
        if prediction == "Healthy":

            health_score = 95

            recommendation = (
                "Machine operating normally"
            )

        elif prediction == "Warning":

            health_score = 65

            recommendation = (
                "Schedule maintenance soon"
            )

        else:

            health_score = 30

            recommendation = (
                "Immediate maintenance required"
            )


        # JSON response
        response = {

            "temperature": temperature,

            "vibration": vibration,

            "current": current,

            "rpm": rpm,

            "prediction": prediction,

            "health_score": health_score,

            "recommendation": recommendation
        }


        return jsonify(response)


    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )