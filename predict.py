import joblib
import pandas as pd

model = joblib.load("employee_absenteeism_model.pkl")

sample_employee = pd.DataFrame([{
    "Age": 32,
    "Workload_Average_Day": 240,
    "Distance_From_Residence": 12,
    "Education": "Graduate",
    "Children": 1,
    "Social_Drinker": "No",
    "Social_Smoker": "No",
    "BMI": 24.5,
    "Transportation_Expense": 160,
    "Month_Value": 6
}])

prediction = model.predict(sample_employee)[0]
probability = model.predict_proba(sample_employee)[0][1]

print("Employee Absenteeism Prediction")
print("--------------------------------")
print("Prediction:", "Absent" if prediction == 1 else "Not Absent")
print(f"Absenteeism Probability: {probability:.2%}")
