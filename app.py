from flask import Flask, render_template, request
import joblib

model = joblib.load("text_model.joblib")
vectorizer = joblib.load("vectorizer.joblib")

app = Flask(__name__)

@app.route("/", methods=["GET","POST"])
def home():

    prediction = None
    probability = None

    if request.method == "POST":

        text = request.form["text"]

        text_vector = vectorizer.transform([text])

        pred = model.predict(text_vector)[0]

        prob = model.predict_proba(text_vector).max()

        prediction = pred
        probability = round(prob*100,2)

    return render_template(
        "index.html",
        prediction=prediction,
        probability=probability
    )

if __name__ == "__main__":
    app.run(debug=True)