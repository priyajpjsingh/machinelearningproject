from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = joblib.load('model/liquidity_model.pkl')
scaler = joblib.load('model/scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = [
            float(request.form['1h']),
            float(request.form['24h']),
            float(request.form['7d']),
            float(request.form['volume']),
            float(request.form['mkt_cap']),
        ]
        input_scaled = scaler.transform([input_data])
        prediction = model.predict(input_scaled)[0]
        return render_template('index.html', prediction_text=f'Predicted Price: ${prediction:.2f}')
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == '__main__':
    app.run(port=8080, debug=True)
