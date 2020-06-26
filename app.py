# Importing essential libraries
from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the Multinomial Naive Bayes model and CountVectorizer object from disk
@app.route('/')
def word_divide_char(inputs):
    characters = []
    for i in inputs:
        characters.append(i)
    return characters

filename = 'pswd_lg_modle.pkl'
classifier = pickle.load(open(filename, 'rb'))
cv = pickle.load(open('pswd-vect.pkl','rb'))

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = cv.transform(data).toarray()
        my_prediction = classifier.predict(vect)
    return render_template('result.html', prediction=my_prediction)


app.run(debug=True)
