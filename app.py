import numpy as np
import pickle
from flask import Flask, render_template, request

app = Flask(__name__)

# Load the trained logistic regression model
model = pickle.load(open('model/logistic.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extracting feature values from the form inputs
        # Note: Ensure these match the encoding used during model training (e.g., 0/1 encoding)
        gender = float(request.form['Gender'])
        married = float(request.form['Married'])
        dependents = float(request.form['Dependents'])
        education = float(request.form['Education'])
        self_employed = float(request.form['Self_Employed'])
        applicant_income = float(request.form['ApplicantIncome'])
        coapplicant_income = float(request.form['CoapplicantIncome'])
        loan_amount = float(request.form['LoanAmount'])
        loan_amount_term = float(request.form['Loan_Amount_Term'])
        credit_history = float(request.form['Credit_History'])
        property_area = float(request.form['Property_Area'])

        # Combine features into an array for prediction
        features = np.array([[
            gender, married, dependents, education, self_employed,
            applicant_income, coapplicant_income, loan_amount,
            loan_amount_term, credit_history, property_area
        ]])

        prediction = model.predict(features)
        
        # Map prediction result to readable text (Adjust based on your target labels)
        output = 'Approved' if prediction[0] == 1 else 'Rejected'

        return render_template('index.html', prediction_text=f'Loan Status: {output}')

    except Exception as e:
        return render_template('index.html', prediction_text=f'Error in processing: {str(e)}')

if __name__ == "__main__":
    app.run(debug=True)
