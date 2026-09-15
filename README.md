Employee Absenteeism Prediction

A machine learning project that predicts employee absenteeism using Python, Scikit-learn, Pandas, and Streamlit.

Project Overview

Employee absenteeism can affect productivity, workforce planning, and business operations. This project uses machine learning to predict whether an employee is likely to be absent based on different personal, work-related, transportation, and lifestyle factors.

The project uses a Random Forest Classifier to classify employees into two categories:

- "0" - Not Absent
- "1" - Absent

The trained model can be used through a Python prediction script or an interactive Streamlit web application.

Objectives

- Predict employee absenteeism using machine learning.
- Analyze factors associated with absenteeism.
- Provide absenteeism probability for individual employees.
- Support batch predictions using CSV files.
- Provide a simple web interface for making predictions.
- Demonstrate a complete machine learning workflow using Scikit-learn.

Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Joblib
- Streamlit

Machine Learning Algorithm

The project uses a:

Random Forest Classifier

Model configuration:

- Number of estimators: 200
- Random state: 42
- Class weight: Balanced

Data Preprocessing

The project uses a Scikit-learn preprocessing pipeline.

Numerical Features

Numerical features are standardized using:

"StandardScaler"

Numerical features include:

- Age
- Workload_Average_Day
- Distance_From_Residence
- Children
- BMI
- Transportation_Expense
- Month_Value

Categorical Features

Categorical features are converted into numerical form using:

"OneHotEncoder"

Categorical features include:

- Education
- Social_Drinker
- Social_Smoker

The preprocessing and machine learning model are combined into a single Scikit-learn Pipeline.

Dataset

The project contains a synthetic dataset with 500 employee records.

Input Features

Feature| Description
Age| Employee age
Workload_Average_Day| Average daily workload
Distance_From_Residence| Distance between residence and workplace
Education| Employee education level
Children| Number of dependent children
Social_Drinker| Whether the employee is a social drinker
Social_Smoker| Whether the employee is a social smoker
BMI| Body Mass Index
Transportation_Expense| Transportation expense
Month_Value| Month represented as a numerical value

Target Variable

"Absenteeism"

- "0" = Not Absent
- "1" = Absent

The dataset contains 250 records for each class.

Model Performance

The dataset is divided into training and testing sets using an 80:20 split with stratification.

The Random Forest model achieves approximately:

Test Accuracy: 74%

Confusion Matrix

The test-set confusion matrix is:

| Predicted Not Absent| Predicted Absent
Actual Not Absent| 39| 11
Actual Absent| 15| 35

This gives:

- Not Absent Precision: 72.22%
- Not Absent Recall: 78.00%
- Absent Precision: 76.09%
- Absent Recall: 70.00%
- Overall Accuracy: 74.00%

Project Structure

Employee_Absenteeism_Prediction_Sklearn/
│
├── app.py
├── predict.py
├── train_model.py
├── employee_absenteeism_model.pkl
├── employee_absenteeism_confusion_matrix.png
├── requirements.txt
├── README.md
│
└── data/
    └── employee_absenteeism.csv

File Description

"train_model.py"

Creates the synthetic employee dataset, prepares the data, trains the Random Forest model, evaluates the model, saves the trained model, and generates the confusion matrix.

"predict.py"

Loads the trained model and performs a prediction for a sample employee.

It displays:

- Prediction result
- Absenteeism probability

"app.py"

Runs the Streamlit web application.

The application provides three main sections:

1. Single Employee Evaluation
2. Batch Prediction using CSV
3. Model Performance and Insights

"employee_absenteeism_model.pkl"

Serialized Scikit-learn model saved using Joblib.

"employee_absenteeism_confusion_matrix.png"

Confusion matrix generated from the model's test predictions.

"employee_absenteeism.csv"

Synthetic dataset containing employee information and absenteeism labels.

"requirements.txt"

Contains the Python libraries required to run the project.

Installation

Clone the repository:

git clone <your-repository-url>
cd Employee_Absenteeism_Prediction_Sklearn

Install the required libraries:

pip install -r requirements.txt

Train the Model

To generate the dataset and train the model:

python train_model.py

This creates:

data/employee_absenteeism.csv
employee_absenteeism_model.pkl
employee_absenteeism_confusion_matrix.png

Make a Prediction

Run:

python predict.py

The program loads the trained model and predicts absenteeism for the sample employee.

Run the Streamlit Application

Start the web application using:

streamlit run app.py

The application allows users to enter employee information and obtain an absenteeism prediction.

Batch Prediction

The Streamlit application also supports CSV-based batch prediction.

Users can upload a CSV containing the required employee features. The application generates:

- Predicted absenteeism status
- Absenteeism probability
- Total employee count
- Predicted absent count
- Predicted not-absent count
- Average absenteeism risk

The prediction results can be exported as a CSV file.

Machine Learning Workflow

Employee Dataset
       |
       v
Data Preparation
       |
       v
Feature Selection
       |
       v
Train-Test Split
       |
       v
Data Preprocessing
       |
       +--------------------+
       |                    |
       v                    v
Numerical Features     Categorical Features
StandardScaler         OneHotEncoder
       |                    |
       +---------+----------+
                 |
                 v
        Random Forest Classifier
                 |
                 v
          Model Evaluation
                 |
                 v
       Prediction and Probability

Example Prediction

The sample employee used in "predict.py" has:

Age: 32
Workload Average/Day: 240
Distance From Residence: 12 km
Education: Graduate
Children: 1
Social Drinker: No
Social Smoker: No
BMI: 24.5
Transportation Expense: 160
Month: 6

The trained model uses these values to predict whether the employee is likely to be absent and provides the estimated probability.

Important Note

This project uses a synthetic dataset created for academic and educational purposes. The predictions should not be treated as real-world HR, medical, or employment decisions.

Future Improvements

- Use a larger real-world dataset.
- Compare Random Forest with Logistic Regression, Decision Tree, XGBoost, and other algorithms.
- Perform hyperparameter tuning.
- Add cross-validation.
- Add feature importance visualization.
- Improve model interpretability.
- Add additional evaluation metrics.
- Deploy the application online.
- Add database support for storing prediction history.

License

This project is intended for educational and academic purposes.# Employee_Absenteeism_Prediction
