from flask import Flask, request, jsonify

app = Flask(__name__)


# Home page
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "running",
        "message": "AI Predictive Maintenance Flask API is working",
        "endpoint": "/predict"
    })


# Prediction API
@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    if not data:
        return jsonify({
            "error": "No JSON data received"
        }), 400

    # Get sensor values
    temperature = float(data.get("temperature", 50))
    vibration = float(data.get("vibration", 2))
    current = float(data.get("current", 5))
    rpm = float(data.get("rpm", 1500))

    # Simple maintenance logic
    if temperature >= 90 or vibration >= 7:
        status = "FAILURE"
        recommendation = "Immediate maintenance required"
        confidence = 95

    elif temperature >= 70 or vibration >= 4:
        status = "WARNING"
        recommendation = "Schedule maintenance"
        confidence = 85

    else:
        status = "NORMAL"
        recommendation = "Machine operating normally"
        confidence = 95

    return jsonify({
        "temperature": temperature,
        "vibration": vibration,
        "current": current,
        "rpm": rpm,
        "status": status,
        "confidence": confidence,
        "recommendation": recommendation
    })


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )