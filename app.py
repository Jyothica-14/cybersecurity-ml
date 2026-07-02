from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
with open("random_forest_model.pkl", "rb") as file:
    model = pickle.load(file)

# Load attack encoder (if you created it)
try:
    with open("attack_encoder.pkl", "rb") as file:
        attack_encoder = pickle.load(file)
except:
    attack_encoder = None


# ================= HOME =================
@app.route("/")
def home():
    return render_template("index.html")


# ================= REGISTER =================
@app.route("/register", methods=["GET", "POST"])
def register():

    if request.method == "POST":
        # No database used.
        # After registration, go to Login page.
        return render_template("login.html")

    return render_template("register.html")


# ================= LOGIN =================
@app.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "POST":
        # No authentication.
        # Directly open prediction page.
        return render_template("predict.html")

    return render_template("login.html")


# ================= PREDICT =================
@app.route("/predict", methods=["GET", "POST"])
def predict():

    if request.method == "GET":
        return render_template("predict.html")

    try:

        timestamp = float(request.form["timestamp"])
        src_ip = float(request.form["src_ip"])
        dst_ip = float(request.form["dst_ip"])
        src_port = float(request.form["src_port"])
        dst_port = float(request.form["dst_port"])
        protocol = float(request.form["protocol"])
        bytes_sent = float(request.form["bytes_sent"])
        bytes_received = float(request.form["bytes_received"])
        user_agent = float(request.form["user_agent"])
        url = float(request.form["url"])
        is_internal = float(request.form["is_internal"])
        label = float(request.form["label"])

        input_data = np.array([[
            timestamp,
            src_ip,
            dst_ip,
            src_port,
            dst_port,
            protocol,
            bytes_sent,
            bytes_received,
            user_agent,
            url,
            is_internal,
            label
        ]])

        prediction = model.predict(input_data)[0]

        # Convert prediction number to attack name
        if attack_encoder is not None:
            prediction = attack_encoder.inverse_transform([prediction])[0]

        return render_template(
            "result.html",
            prediction=prediction
        )

    except Exception as e:

        return render_template(
            "result.html",
            prediction=f"Error: {e}"
        )


if __name__ == "__main__":
    app.run(debug=True)