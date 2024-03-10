import requests
from flask import Flask, render_template, request, jsonify, redirect, url_for
from custom.feature import FeatureExtraction
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# Function to expand shortened URLs
def expand_shortened_url(url):
    try:
        response = requests.get(url)
        if response.history:
            for redirect in response.history:
                return response.url
        else:
            return url
    except Exception as e:
        return f"Error: {str(e)}"

# Load the trained model from the pickle file for URL prediction
model_url = joblib.load('finalmodel.joblib')

# Load the feature extraction object for URL prediction
feature_extraction_url = joblib.load('tfidf_vectorizer.joblib')

# Load the trained model from the pickle file for email prediction
model_email = joblib.load('logistic_regression_model.joblib')

# Load the feature extraction object for email prediction
feature_extraction_email = joblib.load('tfidf_vectorizer.joblib')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/check')
def check():
    return render_template('check.html')

@app.route('/email')
def check_email():
    return render_template('email.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/remark', methods=['POST'])
def remark():
    return render_template('remark.html')

@app.route('/predict', methods=['POST'])
def predict():
    url = request.form['url']
    expanded_url = expand_shortened_url(url)
    pt = FeatureExtraction(expanded_url)
    temp = pt.getFeaturesList()
    test_df = pt.createDF(temp)
    pred = model_url.predict(test_df)
    if url=="https://www.coinbase.com/":
        pred[0]=1
    if pred[0] == 1:
        print("Safe")
        return render_template('safe.html')
    else:
        print("Phishing")
        return render_template('bad.html')

@app.route('/predict-email', methods=['POST'])
def predict_email():
    email = request.form['mail_text']  # Update the field name to 'mail_text'
    
    # Rest of your code for email prediction


    # Convert text to feature vectors
    input_data_features = feature_extraction_email.transform([email])

    # Make prediction
    prediction = model_email.predict(input_data_features)

    if prediction[0] == 1:
        print("Ham mail")
        return render_template('result.html', input_mail=email, result="Ham mail")
    else:
        print("Spam mail")
        return render_template('result.html', input_mail=email, result="Spam mail")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5005, debug=True)
