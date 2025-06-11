from flask import Flask, render_template, request #render template is used to access the index file under templates folder
#request sends a request to index.html file i.e ask for values
import pickle
import numpy as np
import os

app = Flask(__name__) #initialize our flask app

#importing model
with open('house_price_prediction.pkl','rb') as f:
    model = pickle.load(f)

#define the routes for flask app
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    #add all the feaatures of your dataset
    #column names with their dtypes
    features = [
        float(request.form['CRIM']), #every time user gives input it is requested over here
        float(request.form['ZN']),
        float(request.form['INDUS']),
        float(request.form['CHAS']),
        float(request.form['NOX']),
        float(request.form['RM']),
        float(request.form['AGE']),
        float(request.form['DIS']),
        float(request.form['RAD']),
        float(request.form['TAX']),
        float(request.form['PTRATIO']),
        float(request.form['B']),
        float(request.form['LSTAT'])
    ]

    #convert the features into numpy array
    features_array = np.array([features])

    prediction = model.predict(features_array) #predict the value using the model
    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text = f"predicted price of house is {output}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True) #to deploy on render bind Flask app to 0.0.0.0 for Render deployment