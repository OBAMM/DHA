from flask import Flask, jsonify, request, render_template
import json
import numpy as np
import pickle


with open("model.pkl", "rb") as f:
    model = pickle.load(f)

app = Flask(__name__)


@app.route('/', methods=["GET", "POST"])
def index():
    pred = ""
    if request.method == "POST":
        temperature = request.form["temperature"]
        mass_ratio = request.form["mass_ratio"]
        pH = request.form["pH"]
        methanol = request.form["methanol"]
        hexane = request.form["hexane"]
        process_time = request.form["process_time"]
        stiring = request.form["stiring"]
        batch_number = request.form["batch_number"]
        X = np.array([[float(temperature), float(mass_ratio), float(pH), float(methanol), float(hexane), float(process_time), float(stiring), float(batch_number)]])
        prediction = model.predict(X)
        outcome = "high"
        if prediction == 0:
            outcome = "low"
    return render_template("index.html", pred=outcome)


if __name__ == "__main__":
    app.run(debug=True, host='127.0.0.1', port=5000)
