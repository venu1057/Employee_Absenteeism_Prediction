# Employee Absenteeism Prediction

A beginner-friendly machine learning project using Python and Scikit-learn.

## Objective
Predict whether an employee is likely to be absent based on work and personal factors.

## Algorithm
Random Forest Classifier

## Features
- Age
- Workload Average/Day
- Distance From Residence
- Education
- Children
- Social Drink
- Social Smoker
- Body Mass Index
- Transportation Expense
- Month Value

## Target
- Absenteeism: 0 = Not Absent, 1 = Absent

## Run
```bash
pip install -r requirements.txt
python train_model.py
python predict.py
streamlit run app.py
```

The dataset is synthetic and intended for academic/educational use only.
