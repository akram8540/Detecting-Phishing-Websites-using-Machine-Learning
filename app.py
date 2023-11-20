from flask import Flask, request, render_template
import numpy as np
import pickle
from feature import FeatureExtraction

app = Flask(__name__)

# Load the Gradient Boosting Classifier (gbc) model
try:
    with open("pickle/model.pkl", "rb") as model_file:
        gbc = pickle.load(model_file)
except Exception as e:
    print(f"Error loading the model: {str(e)}")
    gbc = None

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        url = request.form["url"]
        if gbc:
            obj = FeatureExtraction(url)
            x = np.array(obj.getFeaturesList()).reshape(1, 30)

            y_pred = gbc.predict(x)[0]
            y_pro_phishing = gbc.predict_proba(x)[0, 0]
            y_pro_non_phishing = gbc.predict_proba(x)[0, 1]

            # Determine if it's safe or unsafe based on the prediction
            if y_pred == 1:
                pred = "It is {0:.2f}% safe to go".format(y_pro_phishing * 100)
            else:
                pred = "It is {0:.2f}% unsafe to go".format(y_pro_non_phishing * 100)

            return render_template('index.html', xx=round(y_pro_non_phishing, 2), url=url, prediction=pred)
        else:
            return "Error: Model not loaded. Please check the model file."

    return render_template("index.html", xx=-1)

if __name__ == "__main__":
    app.run(debug=True)
