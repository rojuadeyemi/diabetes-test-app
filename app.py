import os
import pickle
from flask import Flask, request, render_template
import pandas as pd
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from datetime import datetime
from utility.utility_functions import load_model
from utility.patient_details import categories, test_result

app = Flask(__name__)

# Database config
uri = os.getenv("SQLALCHEMY_DATABASE_URI")

app.config['SQLALCHEMY_DATABASE_URI'] = uri
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
migrate = Migrate(app, db)

# Load model
model = load_model("model/cv_rf.pkl")

# Load columns
with open("column_names.pkl", "rb") as f:
    columns = pickle.load(f)

placeholders, info = categories()

class PatientInfo(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    LastName = db.Column(db.String(50), nullable=False)
    OtherNames = db.Column(db.String(50), nullable=False)
    Age = db.Column(db.String(10), nullable=False)
    Gender = db.Column(db.String(6), nullable=False)
    EmailAddress = db.Column(db.String(100))
    PhoneNumber = db.Column(db.String(16), nullable=False)
    Status = db.Column(db.String(10), nullable=False)
    Date = db.Column(db.DateTime, default=datetime.now())
    Doctor = db.Column(db.String(100), nullable=False)

@app.route('/')
def home():
    return render_template('index.html', info=info, placeholders=placeholders)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract feature names 
        feature_names = [feature for factor in info for feature in info[factor]] 

        # Gather feature values from the form 
        features = [request.form.get(feature) for feature in feature_names] 
        input_data = pd.DataFrame([features], columns=columns)

        # Cast numeric
        input_data = input_data.apply(pd.to_numeric, errors='ignore')

        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        text = test_result(prediction, probability)

        new_patient = PatientInfo(
            LastName=request.form.get('surname', '').title(),
            OtherNames=request.form.get('otherNames', '').title(),
            Age=request.form.get('Age', ''),
            Gender=request.form.get('Gender', ''),
            PhoneNumber=request.form.get('patient_phone', ''),
            EmailAddress=request.form.get('email'),
            Doctor=request.form.get('doctor'),
            Status=text["status"]
        )

        db.session.add(new_patient)
        db.session.commit()

        return render_template('test_result.html', text=text)

    except Exception as e:
        return str(e), 500

if __name__ == "__main__":
    app.run()

