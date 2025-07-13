from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load the trained pipeline (vectorizer + model)
with open("spam_classifier.pkl", "rb") as f:
    clf = pickle.load(f)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    email_text = request.form['email_text']
    if not email_text.strip():
        return render_template('index.html', prediction="Please enter some email text.")

    # Predict directly using the pipeline
    prediction = clf.predict([email_text])[0]
    result = "Spam" if prediction == 1 else "Not Spam"

    return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
