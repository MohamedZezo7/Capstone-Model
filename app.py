import numpy as np
from flask import Flask, request, render_template
from feature import FeatureExtraction  # type: ignore # Assuming this module correctly handles feature extraction
import pickle, joblib
from sklearn.ensemble import GradientBoostingClassifier



# Flask app initialization
flask_app = Flask(__name__)



# Ensure this import if you're checking the instance
'''file = open("model.pkl", "rb")
gbc = pickle.load(file)
file.close()'''

try:
    gbc = joblib.load("model.pkl")
    joblib.dump(gbc, "model.pkl")
    print("Model re-saved successfully.")
except FileNotFoundError:
    print("The original model file was not found.")
except Exception as e:
    print(f"An error occurred: {e}")


# Flask routes
@flask_app.route("/")
def home():
    return render_template("chat.html")

@flask_app.route("/predict_route", methods=["POST"])
def predict_route():
    url = request.form["url"]
    obj = FeatureExtraction(url)
    x = np.array(obj.getFeaturesList()).reshape(1, 30)

    print(type(gbc))  # Debugging: Check the type of gbc

    if isinstance(gbc, GradientBoostingClassifier):  # Ensures gbc is the expected model instance
        y_pred = gbc.predict(x)[0]
        print('='*30)
        print(y_pred)
        print('='*30)
        y_pro_phishing = gbc.predict_proba(x)[0, 0]
        print('='*30)
        print(y_pro_phishing)
        print('='*30)
        y_pro_non_phishing = gbc.predict_proba(x)[0, 1]
        y_pro_non_phishing = 1 - y_pro_non_phishing
        print('='*30)
        print(y_pro_non_phishing)
        print('='*30)
        if y_pred == 1:
            prediction = "It's Safe"
        else:
            prediction = "It's Not Safe"
        # prediction = f"It is {y_pro_phishing * 100:.2f}% safe to go."
        print('='*30)
        print(prediction)
        print('='*30)
    else:
        prediction =  "Error: Model is not loaded correctly"

    return render_template("chat.html", prediction_text=f"{prediction}".title())

# Main function
if __name__ == "__main__":
    flask_app.run(debug=True)
    


