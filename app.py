import pickle
import jsonify
import requests
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from flask import Flask, render_template, request
df = pd.read_csv('Test.csv')
data = df.drop(['Transaction ID', 'Time', 'Class'], axis = 1)
ss = StandardScaler()
data = pd.DataFrame(ss.fit_transform(data), columns = data.columns)
data = pd.concat([df['Transaction ID'], data], axis = 1)
app = Flask(__name__, template_folder = 'Templates')
model = pickle.load(open('fraud_detect.pkl', 'rb'))
@app.route('/', methods = ['GET'])
def Home():    
    return render_template('home.html')
standard_to = StandardScaler()
@app.route('/predict', methods = ['POST'])
def predict():
    if request.method == 'POST':
        txn_id = int(request.form['aa'])
        filtered_data = data[data['Transaction ID'] == txn_id]
        prediction = model.predict_proba(filtered_data.iloc[:, 1:])[0][1]
        prediction = np.round(prediction * 100, 2)
        return render_template('predict.html', prediction_text = 'The probability of this transaction being fraudulent is {}%.'.format(prediction))
    else:
        return render_template('home.html')
if __name__=="__main__":
    #app.run(host = '0.0.0.0', port = 8080)
    app.run(debug = True)