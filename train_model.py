import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

np.random.seed(42)

os.makedirs("data", exist_ok=True)

n = 500
df = pd.DataFrame({
    "Age": np.random.randint(20, 61, n),
    "Workload_Average_Day": np.random.randint(100, 380, n),
    "Distance_From_Residence": np.random.randint(1, 55, n),
    "Education": np.random.choice(["High School", "Diploma", "Graduate", "Postgraduate"], n),
    "Children": np.random.randint(0, 4, n),
    "Social_Drinker": np.random.choice(["Yes", "No"], n, p=[0.35, 0.65]),
    "Social_Smoker": np.random.choice(["Yes", "No"], n, p=[0.18, 0.82]),
    "BMI": np.round(np.random.uniform(18, 36, n), 1),
    "Transportation_Expense": np.random.randint(50, 320, n),
    "Month_Value": np.random.randint(1, 13, n)
})

risk = (
    0.015 * df["Age"]
    + 0.004 * df["Workload_Average_Day"]
    + 0.018 * df["Distance_From_Residence"]
    + 0.25 * (df["Social_Drinker"] == "Yes")
    + 0.35 * (df["Social_Smoker"] == "Yes")
    + 0.08 * df["Children"]
    + 0.003 * df["Transportation_Expense"]
    + 0.15 * (df["BMI"] > 30)
    + np.random.normal(0, 0.7, n)
)

threshold = np.median(risk)
df["Absenteeism"] = (risk > threshold).astype(int)

dataset_path = "data/employee_absenteeism.csv"
df.to_csv(dataset_path, index=False)

X = df.drop(columns=["Absenteeism"])
y = df["Absenteeism"]

categorical_features = ["Education", "Social_Drinker", "Social_Smoker"]
numeric_features = [c for c in X.columns if c not in categorical_features]

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    ))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model.fit(X_train, y_train)
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:\n")
print(classification_report(y_test, predictions))

joblib.dump(model, "employee_absenteeism_model.pkl")

cm = confusion_matrix(y_test, predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Absent", "Absent"])
disp.plot()
plt.title("Employee Absenteeism Prediction - Confusion Matrix")
plt.tight_layout()
plt.savefig("employee_absenteeism_confusion_matrix.png")
plt.close()

print("\nModel saved as employee_absenteeism_model.pkl")
print("Dataset saved as data/employee_absenteeism.csv")
