import numpy as np
import pandas as pd
import pickle
from flask import Flask, request, jsonify, render_template


app = Flask(__name__)
scaler = pickle.load(open('StandardScaler.pkl', 'rb'))
model = pickle.load(open('LogisticRegression.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods =['POST'])
def predict():
    features = [x for x in request.form.values()]
    features = np.array(features).reshape(1,len(features))
    scaled_features = scaler.transform(features)
    prediction = model.predict(scaled_features)
    
    if prediction[0] == 1:
        result = "The person's vital sign shows that he/she is diabetic"
        return render_template('index1.html',prediction_text = result)
    else:
        result = "The Person doesn't have sign of Diabetes"
        return render_template('index2.html',prediction_text = result)    
    

if __name__ == "__main__":
    app.run(debug=True)