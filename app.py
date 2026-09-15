import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from PIL import Image

# Page Configuration
st.set_page_config(
    page_title="Employee Absenteeism Predictor",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for rich, polished aesthetics
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 26px 30px;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.2);
    }
    
    .badge-pill {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        background: rgba(99, 102, 241, 0.15);
        color: #818CF8;
        border: 1px solid rgba(99, 102, 241, 0.3);
        margin-bottom: 10px;
    }
    
    .card {
        background: #1E293B;
        border: 1px solid #334155;
        border-radius: 14px;
        padding: 20px;
        margin-bottom: 16px;
    }
    
    .result-card-safe {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(5, 150, 105, 0.05) 100%);
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 16px;
        padding: 24px;
        text-align: center;
    }
    
    .result-card-risk {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.12) 0%, rgba(220, 38, 38, 0.05) 100%);
        border: 1px solid rgba(239, 68, 68, 0.35);
        border-radius: 16px;
        padding: 24px;
        text-align: center;
    }
    
    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.5px;
    }
    
    .recommendation-box {
        background: rgba(30, 41, 59, 0.7);
        border-left: 4px solid #6366F1;
        padding: 14px 18px;
        border-radius: 0 10px 10px 0;
        margin-top: 14px;
    }
</style>
""", unsafe_allow_html=True)

# Load Model
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "employee_absenteeism_model.pkl")
    if not os.path.exists(model_path):
        model_path = "employee_absenteeism_model.pkl"
    return joblib.load(model_path)

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Header
st.markdown("""
<div class="main-header">
    <div class="badge-pill">AI-Powered Workforce Analytics</div>
    <h1 style="color: #F8FAFC; margin: 0; font-weight: 800; font-size: 2.2rem;">🏢 Employee Absenteeism Prediction</h1>
    <p style="color: #94A3B8; margin-top: 8px; margin-bottom: 0; font-size: 1.05rem;">
        Predict absenteeism risk and identify critical workforce health and commute factors using machine learning.
    </p>
</div>
""", unsafe_allow_html=True)

tabs = st.tabs(["🎯 Single Employee Evaluation", "📁 Batch Prediction (CSV)", "📊 Model Performance & Insights"])

# Month name mapping helper
MONTH_NAMES = {
    1: "January", 2: "February", 3: "March", 4: "April",
    5: "May", 6: "June", 7: "July", 8: "August",
    9: "September", 10: "October", 11: "November", 12: "December"
}

# --- TAB 1: Single Employee Evaluation ---
with tabs[0]:
    st.subheader("Employee Profile & Work Parameters")
    
    # Preset Quick Loaders
    preset_cols = st.columns([1, 1, 1, 3])
    with preset_cols[0]:
        preset_low = st.button("🌱 Load Low-Risk Preset", use_container_width=True)
    with preset_cols[1]:
        preset_high = st.button("⚠️ Load High-Risk Preset", use_container_width=True)
    with preset_cols[2]:
        preset_sample = st.button("📋 Load Sample Preset", use_container_width=True)

    if preset_low:
        st.session_state["age"] = 28
        st.session_state["workload"] = 180
        st.session_state["distance"] = 8
        st.session_state["education"] = "Graduate"
        st.session_state["children"] = 0
        st.session_state["drinker"] = "No"
        st.session_state["smoker"] = "No"
        st.session_state["bmi"] = 22.0
        st.session_state["transport"] = 120
        st.session_state["month"] = 4
    elif preset_high:
        st.session_state["age"] = 52
        st.session_state["workload"] = 350
        st.session_state["distance"] = 48
        st.session_state["education"] = "High School"
        st.session_state["children"] = 3
        st.session_state["drinker"] = "Yes"
        st.session_state["smoker"] = "Yes"
        st.session_state["bmi"] = 32.5
        st.session_state["transport"] = 290
        st.session_state["month"] = 11
    elif preset_sample:
        st.session_state["age"] = 32
        st.session_state["workload"] = 240
        st.session_state["distance"] = 12
        st.session_state["education"] = "Graduate"
        st.session_state["children"] = 1
        st.session_state["drinker"] = "No"
        st.session_state["smoker"] = "No"
        st.session_state["bmi"] = 24.5
        st.session_state["transport"] = 160
        st.session_state["month"] = 6

    # Input Form organized in two clean columns
    col_left, col_right = st.columns(2, gap="large")

    with col_left:
        st.markdown("#### 👤 Personal & Health Profile")
        age = st.slider("Age (years)", 18, 65, value=st.session_state.get("age", 32), key="input_age")
        bmi = st.slider("Body Mass Index (BMI)", 15.0, 40.0, value=float(st.session_state.get("bmi", 24.5)), step=0.1, key="input_bmi",
                        help="BMI > 30 is categorized as obese and correlates with higher absenteeism risk.")
        education = st.selectbox(
            "Education Level",
            ["High School", "Diploma", "Graduate", "Postgraduate"],
            index=["High School", "Diploma", "Graduate", "Postgraduate"].index(st.session_state.get("education", "Graduate")),
            key="input_education"
        )
        children = st.number_input("Number of Dependent Children", 0, 6, value=int(st.session_state.get("children", 1)), key="input_children")
        
        c_sub1, c_sub2 = st.columns(2)
        with c_sub1:
            social_drinker = st.selectbox("Social Drinker?", ["No", "Yes"], 
                                          index=["No", "Yes"].index(st.session_state.get("drinker", "No")), 
                                          key="input_drinker")
        with c_sub2:
            social_smoker = st.selectbox("Social Smoker?", ["No", "Yes"], 
                                         index=["No", "Yes"].index(st.session_state.get("smoker", "No")), 
                                         key="input_smoker")

    with col_right:
        st.markdown("#### 💼 Workload & Commute Factors")
        workload = st.slider("Workload Average / Day", 80, 400, value=int(st.session_state.get("workload", 240)), step=5, key="input_workload",
                             help="Average daily workload units or expected task velocity.")
        distance = st.slider("Distance From Residence (km)", 1, 60, value=int(st.session_state.get("distance", 12)), key="input_distance",
                             help="Commute distance from employee home to work premises.")
        transport = st.slider("Monthly Transportation Expense ($)", 30, 400, value=int(st.session_state.get("transport", 160)), step=10, key="input_transport")
        
        month = st.selectbox(
            "Evaluation Month",
            list(MONTH_NAMES.keys()),
            format_func=lambda m: f"{m} - {MONTH_NAMES[m]}",
            index=list(MONTH_NAMES.keys()).index(st.session_state.get("month", 6)),
            key="input_month"
        )

    st.markdown("---")
    
    # Run Prediction
    predict_btn = st.button("🚀 Analyze Absenteeism Risk", type="primary", use_container_width=True)
    
    # Prepare DataFrame
    employee_input = pd.DataFrame([{
        "Age": age,
        "Workload_Average_Day": workload,
        "Distance_From_Residence": distance,
        "Education": education,
        "Children": children,
        "Social_Drinker": social_drinker,
        "Social_Smoker": social_smoker,
        "BMI": bmi,
        "Transportation_Expense": transport,
        "Month_Value": month
    }])

    # Automatic evaluation on submit or load
    pred = model.predict(employee_input)[0]
    prob = model.predict_proba(employee_input)[0][1]

    st.markdown("### 📋 Prediction Outcome")
    res_col1, res_col2 = st.columns([1.2, 1.8], gap="medium")

    with res_col1:
        if pred == 1:
            st.markdown(f"""
            <div class="result-card-risk">
                <span style="font-size: 2.4rem;">⚠️</span>
                <div style="color: #F87171; font-weight: 700; font-size: 1.1rem; text-transform: uppercase; letter-spacing: 1px; margin-top: 6px;">
                    High Risk of Absence
                </div>
                <div class="metric-value" style="color: #EF4444;">
                    {prob:.1%}
                </div>
                <p style="color: #94A3B8; font-size: 0.9rem; margin: 0;">Predicted Likelihood of Absenteeism</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-card-safe">
                <span style="font-size: 2.4rem;">✅</span>
                <div style="color: #34D399; font-weight: 700; font-size: 1.1rem; text-transform: uppercase; letter-spacing: 1px; margin-top: 6px;">
                    Low Risk / Regular Attendance
                </div>
                <div class="metric-value" style="color: #10B981;">
                    {prob:.1%}
                </div>
                <p style="color: #94A3B8; font-size: 0.9rem; margin: 0;">Predicted Likelihood of Absenteeism</p>
            </div>
            """, unsafe_allow_html=True)

        st.progress(float(prob))

    with res_col2:
        st.markdown("#### 🔍 Contributing Risk Analysis")
        risk_flags = []
        if workload > 280:
            risk_flags.append(("Elevated Workload", f"Workload ({workload}) is above average, which often causes fatigue."))
        if distance > 30:
            risk_flags.append(("Long Commute", f"Commuting {distance} km increases transit fatigue and schedule friction."))
        if social_smoker == "Yes":
            risk_flags.append(("Smoking Lifestyle", "Social smoking is positively associated with health-related downtime."))
        if social_drinker == "Yes":
            risk_flags.append(("Social Drinking", "Social drinking correlates with periodic post-weekend attendance disruption."))
        if bmi >= 30:
            risk_flags.append(("BMI Indicator (Obesity range)", f"BMI of {bmi:.1f} may indicate susceptibility to health complaints."))
        if children >= 3:
            risk_flags.append(("Family Care Demands", f"{children} dependent children may require unexpected family leave."))

        if risk_flags:
            for title, desc in risk_flags:
                st.warning(f"**{title}**: {desc}")
        else:
            st.success("🎉 **Balanced Profile**: No significant high-risk work or lifestyle triggers detected.")

        st.markdown(f"""
        <div class="recommendation-box">
            <strong style="color: #818CF8;">HR / Management Recommendation:</strong><br>
            <span style="color: #CBD5E1; font-size: 0.92rem;">
                {'Consider proactive 1-on-1 check-ins, hybrid/remote commute flexibility, or workload redistribution.' if pred == 1 else 'Employee parameters are within sustainable thresholds. Continue regular engagement and wellness practices.'}
            </span>
        </div>
        """, unsafe_allow_html=True)

    with st.expander("🔍 View Raw Transformed Input"):
        st.dataframe(employee_input, use_container_width=True)

# --- TAB 2: Batch Prediction (CSV) ---
with tabs[1]:
    st.subheader("Batch Absenteeism Analysis")
    st.write("Upload a CSV file with employee data or analyze the pre-existing dataset.")

    csv_file = st.file_uploader("Upload Employee Data CSV", type=["csv"])
    
    data_source = None
    if csv_file is not None:
        data_source = pd.read_csv(csv_file)
        st.info(f"Loaded {len(data_source)} records from uploaded file.")
    else:
        sample_csv_path = os.path.join(os.path.dirname(__file__), "data", "employee_absenteeism.csv")
        if os.path.exists(sample_csv_path):
            if st.checkbox("Load baseline dataset (data/employee_absenteeism.csv)", value=True):
                data_source = pd.read_csv(sample_csv_path)
                st.info(f"Loaded {len(data_source)} records from sample dataset.")

    if data_source is not None:
        feature_cols = ["Age", "Workload_Average_Day", "Distance_From_Residence", "Education",
                        "Children", "Social_Drinker", "Social_Smoker", "BMI", 
                        "Transportation_Expense", "Month_Value"]
        
        missing = [c for c in feature_cols if c not in data_source.columns]
        if missing:
            st.error(f"Missing required columns in dataset: {missing}")
        else:
            if st.button("⚡ Run Batch Prediction", type="primary"):
                with st.spinner("Processing batch predictions..."):
                    batch_pred = model.predict(data_source[feature_cols])
                    batch_prob = model.predict_proba(data_source[feature_cols])[:, 1]
                    
                    results_df = data_source.copy()
                    results_df["Predicted_Absenteeism"] = ["Absent" if p == 1 else "Not Absent" for p in batch_pred]
                    results_df["Absent_Probability"] = np.round(batch_prob * 100, 1)

                    # Summary Stats
                    absent_count = sum(batch_pred == 1)
                    not_absent_count = sum(batch_pred == 0)
                    avg_prob = np.mean(batch_prob) * 100

                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Total Employees", len(results_df))
                    m2.metric("Predicted Absent", absent_count, delta=f"{(absent_count/len(results_df))*100:.1f}%")
                    m3.metric("Predicted Not Absent", not_absent_count)
                    m4.metric("Avg Absenteeism Risk", f"{avg_prob:.1f}%")

                    # Filter Options
                    filter_choice = st.radio("Filter Results View:", ["All", "High Risk Only (Absent)", "Low Risk Only (Not Absent)"], horizontal=True)
                    if filter_choice == "High Risk Only (Absent)":
                        display_df = results_df[results_df["Predicted_Absenteeism"] == "Absent"]
                    elif filter_choice == "Low Risk Only (Not Absent)":
                        display_df = results_df[results_df["Predicted_Absenteeism"] == "Not Absent"]
                    else:
                        display_df = results_df

                    st.dataframe(display_df, use_container_width=True)

                    # CSV Download
                    csv_export = results_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Predictions as CSV",
                        data=csv_export,
                        file_name="employee_absenteeism_predictions.csv",
                        mime="text/csv",
                    )

# --- TAB 3: Model Performance & Insights ---
with tabs[2]:
    st.subheader("Model Evaluation & Architecture")
    
    col_info1, col_info2 = st.columns(2)
    with col_info1:
        st.markdown("""
        #### 🤖 Model Specifications
        - **Algorithm**: Random Forest Classifier (`n_estimators=200`, `class_weight='balanced'`)
        - **Preprocessing Pipeline**: 
            - `StandardScaler` on continuous numerical features
            - `OneHotEncoder` on categorical features (`Education`, `Social_Drinker`, `Social_Smoker`)
        - **Evaluation Metric**: Stratified Test Accuracy ~ 74%
        - **Model Persistence**: Serialized with `joblib`
        """)
    
    with col_info2:
        cm_path = os.path.join(os.path.dirname(__file__), "employee_absenteeism_confusion_matrix.png")
        if os.path.exists(cm_path):
            st.image(cm_path, caption="Confusion Matrix on Test Split", use_container_width=True)
        else:
            st.info("Confusion matrix image not found.")

st.caption("Developed for Employee Absenteeism Analytics • Scikit-learn & Streamlit")
