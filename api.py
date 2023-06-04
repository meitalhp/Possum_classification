import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import os

app = Flask(__name__)
rf_model = pickle.load(open('possum_rf_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    features = request.form.getlist('feature')
    
    final_features = [features]
    prediction = rf_model.predict(final_features)[0]
    #risk = rf_model.predict_proba(final_features)[0][1]

    if prediction == 'm':
        output_text = "This possum is male"
    else:
        output_text = "This possum is female"

    return render_template('index.html', prediction_text='{}'.format(output_text))


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    
    app.run(host='0.0.0.0', port=port,debug=True)
