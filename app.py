from flask import Flask, render_template, url_for, request
import pandas as pd
import pickle


from scrape_link import Artdata
import clean_text
from is_url import is_url

tf1 = pickle.load(open("models/tfidf1.pkl", 'rb'))
# rfc = pickle.load(open("models/rfc.pkl", 'rb'))
logc = pickle.load(open("models/logc.pkl", 'rb'))

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    message = request.form['message']
    if is_url(message):
        article = Artdata(message)
        text = article.article_text()
    else:
        text = message
    cleaned_text = clean_text.pipe.transform([text])
    x = tf1.transform(cleaned_text)
    y = logc.predict(x)[0]
    return render_template('result.html', prediction = y)


if __name__ == '__main__':
    app.run(debug=True)
