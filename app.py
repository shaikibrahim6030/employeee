from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

model = joblib.load('salary_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    experience = float(request.form['experience'])
    age = float(request.form['age'])

    prediction = model.predict([[experience, age]])

    return render_template('result.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
