from flask import Flask, request, render_template,redirect, url_for
import pandas as pd
import logging
from utility.performance_report import find_best
from utility.utility_functions import load_model
from utility.patient_details import categories, test_result
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from datetime import datetime

# Initialize Flask app
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///mydatabase.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
migrate = Migrate(app, db)

class PatientInfo(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    LastName = db.Column(db.String(50), nullable=False)
    OtherNames = db.Column(db.String(50), nullable=False)
    Age = db.Column(db.String(10), nullable=False)
    Gender = db.Column(db.String(6), nullable=False)
    EmailAddress = db.Column(db.String(100), nullable=True)
    PhoneNumber = db.Column(db.String(16),nullable=False)
    Status = db.Column(db.String(10), nullable=False)
    Date = db.Column(db.DateTime, default=datetime.now())
    

# Configure logging
logging.basicConfig(
    filename='app.log',  # Log file location
    level=logging.INFO,  # Set log level to INFO or DEBUG for more detail
    format='%(asctime)s - %(levelname)s - %(message)s'  # Log format
)
logger = logging.getLogger()

# Initialize and load the best model
path = "./model"
_, model_path = find_best(path)
model = load_model(model_path)

#Extract the columns from the dataset
columns= pd.read_csv("./processed_data/X_test.csv").columns

# Get feature categories
placeholders, info = categories()

@app.route('/')
def home():
    logger.info("Home page accessed")
    return render_template('index.html', info=info,placeholders=placeholders)

@app.route('/predict', methods=['POST'])
def predict():
    
    try:
        logger.info("Prediction request received")
        
        # Extract feature names
        feature_names = [feature for factor in info for feature in info[factor]]
        
        # Gather feature values from the form
        features = [request.form.get(feature, 'N/A') for feature in feature_names]
        input_data = pd.DataFrame([features], columns=columns)


        # Predict and calculate probability
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]  # Probability of having diabetes

        # Prepare test results
        text = test_result(prediction, probability)

        # Prepare patient info
        patient_info = {
            "Surname": request.form.get('surname', 'N/A').title(),
            "Other Names": request.form.get('otherNames', 'N/A').title(),
            "Age": request.form.get('Age', 'N/A')+" Years",
            "Gender": request.form.get('Gender', 'N/A'),
            "Phone Number": request.form.get('patient_phone', 'N/A'),
            "Email": request.form.get('email', 'NIL')
        }

        #Update the database
        new_patient = PatientInfo(EmailAddress=patient_info["Email"], PhoneNumber=patient_info["Phone Number"],
                                  LastName = patient_info["Surname"],OtherNames = patient_info["Other Names"], 
                                  Age=patient_info["Age"],Gender=patient_info["Gender"],Status=text["status"])
        db.session.add(new_patient)
        db.session.commit()

        return render_template('test_result.html', text=text, patient_info=patient_info)

    except KeyError as e:
        logger.error(f"Missing input for: {e.args[0]}")
        return f"Missing input for: {e.args[0]}", 400
    except ValueError as e:
        logger.error(f"Value error: {str(e)}")
        return str(e), 400
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return str(e), 500

if __name__ == "__main__":
    app.run()

