import pandas as pd
# from werkzeug.wrappers import Request, Response
from flask import Flask, render_template, request, jsonify, redirect
import pickle
import webbrowser
import numpy as np
from threading import Timer

# load model
model = pickle.load(open('D:\Chola Finance\divyam file\model.pkl','rb'))

# app
app = Flask(__name__)

x=[]
# routes
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        hours = request.form.get('hours')       
        prediction = model.predict([[np.array(hours)]])
        # Take the first value of prediction
        x.append(prediction)
        return redirect ('/predict')
    return render_template('home.html')

@app.route('/predict')
def predict():
        return render_template('predict.html' , x = x )

def open_browser():
      webbrowser.open_new('http://127.0.0.1:5000/')

if __name__ == "__main__":
      Timer(1, open_browser).start();
      app.run(port=5000)