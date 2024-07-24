def categories():

    enforce = {
        "Age": ["number","\d+","Only integer input is allowed"],
        "Gender": ["text","Male|Female","Options:Female, Male"],
        "Ethnicity": ["text","African American|Caucasian|Hispanic|Asian", "Options: African American, Caucasian, Hispanic, Asian"],
        "Socioeconomic Status": ["text","High|Middle|Low", "Options: High, Middle, Low"],
        "Education Level": ["text","No formal education|Primary|Secondary|Higher",
                      "Options: No formal education, Primary, Secondary, Higher"],
        "BMI": ["number","^\d+(\.\d+)?$","Only numeric input is allowed"],
        "Smoking": ["text","Yes|No","Options: Yes, No"],
        "Alcohol Consumption": ["number","^(?:[0-19](?:\.\d+)?|1[0-9](?:\.\d+)?|20(?:\.0+)?)$","Enter a value between 0 and 20"],
        "Physical Activity":["number","^(?:[0-9](?:\.\d+)?|10(?:\.0+)?)$","Enter a value between 0 and 10"],
        "Diet Quality": ["number","^(?:[0-9](?:\.\d+)?|10(?:\.0+)?)$","Enter a value between 0 and 10"],
        "Sleep Quality": ["number","^(?:[0-9](?:\.\d+)?|10(?:\.0+)?)$","Enter a value between 0 and 10"],
        "Family History Diabetes": ["text","Yes|No","Options: Yes, No"],
        "Gestational Diabetes": ["text","Yes|No","Options: Yes, No"],
        "Polycystic Ovary Syndrome": ["text","Yes|No","Options: Yes, No"],
        "Previous PreDiabetes": ["text","Yes|No","Options: Yes, No"],
        "Hypertension": ["text","Yes|No","Options: Yes, No"],
        "Systolic Blood Pressure": ["number","^\d+(\.\d+)?$","Only numeric input is allowed"],
        "Diastolic Blood Pressure": ["number","^\d+(\.\d+)?$","Only numeric input is allowed"],
        "Fasting Blood Sugar": ["number","^\d+(\.\d+)?$","Only numeric input is allowed"],
        "HbA1c": ["number","^\d+(\.\d+)?$","Only numeric input is allowed"],
        "Serum Creatinine": ["number","^\d+(\.\d+)?$","Only numeric input is allowed"],
        "BUN Levels": ["number","^\d+(\.\d+)?$","Only numeric input is allowed"],
        "Cholesterol Total": ["number","^\d+(\.\d+)?$","Only numeric input is allowed"],
        "Cholesterol LDL": ["number","^\d+(\.\d+)?$","Only numeric input is allowed"],
        "Cholesterol HDL": ["number","^\d+(\.\d+)?$","Only numeric input is allowed"],
        "Cholesterol Triglycerides": ["number","^\d+(\.\d+)?$","Only numeric input is allowed"],
        "Antihypertensive Medications": ["text","Yes|No","Options: Yes, No"],
        "Statins": ["text","Yes|No","Options: Yes, No"],
        "Antidiabetic Medications": ["text","Yes|No","Options: Yes, No"],
        "Frequent Urination": ["text","Yes|No","Options: Yes, No"],
        "Excessive Thirst": ["text","Yes|No","Options: Yes, No"],
        "Unexplained WeightLoss": ["text","Yes|No","Options: Yes, No"],
        "Fatigue Levels": ["number","^(?:[0-9](?:\.\d+)?|10(?:\.0+)?)$","Enter a value between 0 and 10"],
        "Blurred Vision": ["text","Yes|No","Options: Yes, No"],
        "Slow Healing Sores": ["text","Yes|No","Options: Yes, No"],
        "Tingling Hands & Feet": ["text","Yes|No","Options: Yes, No"],
        "Quality Of Life Score": ["number","^(?:100(?:\.0+)?|[0-9]{1,2}(?:\.\d+)?)$","Enter a value between 0 and 100"],
        "Heavy Metals Exposure": ["text","Yes|No","Options: Yes, No"],
        "Occupational Exposure Chemicals": ["text","Yes|No","Options: Yes, No"],
        "Water Quality": ["text","Yes|No","Options: Yes, No"],
        "Medical Checkups Frequency": ["number","^(?:[0-3](?:\.\d+)?|4(?:\.0+)?)$","Enter a value between 0 and 4"],
        "Medication Adherence": ["number","^(?:[0-9](?:\.\d+)?|10(?:\.0+)?)$","Enter a value between 0 and 10"],
        "Health Literacy": ["number","^(?:[0-9](?:\.\d+)?|10(?:\.0+)?)$","Enter a value between 0 and 10"]
    }


    info= {
            "Demographic Information": [
                "Age",
                "Gender",
                "Ethnicity",
                "Socioeconomic Status",
                "Education Level"
            ],
            "Lifestyle Factors": [
                "BMI",
                "Smoking",
                "Alcohol Consumption",
                "Physical Activity",
                "Diet Quality",
                "Sleep Quality"
            ],
            "Medical History": [
                "Family History Diabetes",
                "Gestational Diabetes",
                "Polycystic Ovary Syndrome",
                "Previous PreDiabetes",
                "Hypertension"
            ],
            "Vital Signs and Lab Results": [
                "Systolic Blood Pressure",
                "Diastolic Blood Pressure",
                "Fasting Blood Sugar",
                "HbA1c",
                "Serum Creatinine",
                "BUN Levels",
                "Cholesterol Total",
                "Cholesterol LDL",
                "Cholesterol HDL",
                "Cholesterol Triglycerides"
            ],
            "Medications": [
                "Antihypertensive Medications",
                "Statins",
                "Antidiabetic Medications"
            ],
            "Symptoms": [
                "Frequent Urination",
                "Excessive Thirst",
                "Unexplained WeightLoss",
                "Fatigue Levels",
                "Blurred Vision",
                "Slow Healing Sores",
                "Tingling Hands & Feet"
            ],
            "Quality of Life and Environmental Factors": [
                "Quality Of Life Score",
                "Heavy Metals Exposure",
                "Occupational Exposure Chemicals",
                "Water Quality"
            ],
            "Healthcare and Adherence": [
                "Medical Checkups Frequency",
                "Medication Adherence",
                "Health Literacy"
            ]
        }
    return enforce,info


def test_result(prediction,probability):
    if prediction == 0:
            status = "Normal"
            remark = f"Test results are within the normal range. It is unlikely that the patient may have diabetes (prob. {probability:.0%})."
    elif prediction == 1:
            status = "Positive"
            remark = f"Test results indicate that the patient may have diabetes (prob. {probability:.0%}). Please schedule an appointment with the patient's healthcare provider for further evaluation and management."

    return {"status": status, "remark": remark}