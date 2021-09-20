import pandas as pd
# from werkzeug.wrappers import Request, Response
from flask import Flask, render_template, request, jsonify, redirect
import pickle
import webbrowser
import numpy as np

# load model
model = pickle.load(open('model.pkl','rb'))

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

if __name__ == "__main__":
        app.run(debug=False, port=int(os.environ.get("PORT", 5000)), host='0.0.0.0')
