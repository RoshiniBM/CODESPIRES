import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
import numpy as np

# --- Reference ranges for context (Used for the 'Degree of Affect' table) ---
NORMAL_RANGES = {
    'Age': 50,         # Arbitrary boundary for "middle-aged" risk
    'BMI': 25.0,       # Healthy weight BMI maximum (kg/m²)
    'Glucose': 100,    # Fasting Glucose normal maximum (mg/dL)
    'Insulin': 15.0,   # Fasting Serum Insulin normal maximum (mu U/mL)
    'BloodPressure': 80, # Diastolic BP normal maximum (mmHg)
    'Hba1c': 5.7,      # HbA1c pre-diabetes cutoff (%)
    'PP_Glucose': 140  # Post-Prandial Glucose normal maximum (mg/dL)
}

# Function to get feature coefficients (Innovation - XAI)
def get_feature_importance(model, features):
    # Retrieve the coefficients (weights) from the trained model
    weights = pd.Series(model.coef_[0], index=features).sort_values(ascending=False)
    return weights

# --- 1. MODEL TRAINING FUNCTION ---
@st.cache_resource 
def train_model():
    try:
        # Load data with the encoding fix
        df = pd.read_csv('patient_data.csv', encoding='utf-8')
    except Exception as e:
        st.error(f"Error loading data: {e}. Please check patient_data.csv format.")
        return None, None, None
    
    # FINAL: Updated features list
    features = ['Age', 'BMI', 'Glucose', 'Insulin', 'BloodPressure', 'Hba1c', 'PP_Glucose']
    
    # Check for missing columns
    if not all(col in df.columns for col in features + ['Diabetes_Risk']):
        st.error("ERROR: CSV file is missing one or more required columns.")
        return None, None, None

    X = df[features]
    y = df['Diabetes_Risk']

    # Split data for training best practice (Technical Implementation)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = LogisticRegression(solver='liblinear') 
    model.fit(X_train, y_train) 
    
    # Calculate accuracy
    accuracy = model.score(X_test, y_test)
    
    # Display accuracy in the sidebar
    st.sidebar.markdown("---")
    st.sidebar.header("📊 Model Performance")
    st.sidebar.markdown(f"**Accuracy on Test Data:** **{accuracy*100:.1f}%**")
    st.sidebar.caption("Based on Logistic Regression.")

    return model, features, df # Return the model, features, and DataFrame


# Get the trained model, features, and data frame
model, features, df = train_model()

# Exit if model failed to load 
if model is None:
    st.stop()
    
# --- 2. STREAMLIT INTERFACE SETUP ---

st.title("🩺 Automated Diabetes Risk Assessment System")
st.markdown("---")
st.header("Patient Lab Report Input")

# Create input fields for the user to enter data
col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input('Age (Years)', min_value=18, max_value=100, value=40)
    bmi = st.number_input('BMI (kg/m²)', min_value=15.0, max_value=50.0, value=25.0, format="%.1f")

with col2:
    glucose = st.number_input('Glucose (mg/dL) - Fasting', min_value=50, max_value=300, value=100)
    # Updated label for clarity
    insulin = st.number_input('Insulin (mu U/mL) - Fasting Serum', min_value=1.0, max_value=80.0, value=10.0, format="%.1f")
    # Post-Prandial Glucose
    pp_glucose = st.number_input('PP Glucose (mg/dL) - After Meal', min_value=70, max_value=400, value=120)

with col3:
    blood_pressure = st.number_input('Blood Pressure (mmHg) - Diastolic', min_value=50, max_value=120, value=75)
    # HbA1c input
    hba1c = st.number_input('HbA1c (%) - Glycated Hemoglobin', min_value=4.0, max_value=12.0, value=5.5, format="%.1f")
    
st.markdown("---")

# --- 3. PREDICTION LOGIC ---

# FIX: Added unique key to resolve StreamlitDuplicateElementId error
if st.button('🔬 Analyze Report', key='analyze_button'):
    # Prepare the input data (MUST match the 'features' list order)
    input_data = np.array([[age, bmi, glucose, insulin, blood_pressure, hba1c, pp_glucose]])
    
    # 1. Predict the probability of high risk (P=1)
    risk_proba = model.predict_proba(input_data)[:, 1][0]
    
    # 2. Determine the diagnosis and color
    if risk_proba >= 0.75: 
        diagnosis = "⚠️ CRITICAL RISK - Consult Immediately"
        color = 'red'
        suggestion = "Recommendation: Urgent consultation with an Endocrinologist and schedule a confirmatory Oral Glucose Tolerance Test (OGTT)."
    elif risk_proba >= 0.45: 
        diagnosis = "🟡 MODERATE RISK - Lifestyle Intervention Needed"
        color = 'orange'
        suggestion = "Recommendation: Recommend lifestyle changes (diet/exercise) and schedule a follow-up lab test within 3 months."
    else:
        diagnosis = "✅ LOW RISK - No immediate risk indication"
        color = 'green'
        suggestion = "Recommendation: Continue healthy lifestyle and routine annual check-ups."

    # --- DISPLAY RESULTS ---
    
    st.subheader("Diagnostic Insight:")
    st.markdown(f"## <span style='color:{color};'>{diagnosis}</span>", unsafe_allow_html=True)
    
    st.subheader("Risk Score & Confidence:")
    st.metric(label="Probability of High Risk", value=f"{risk_proba * 100:.1f} %")

    # Risk Gauge/Progress Bar
    st.progress(risk_proba)
    st.caption("Risk confidence displayed visually.")
    
    # --- Degree of Affect Table (XAI Innovation) ---
    st.subheader("Factor Contribution & Degree of Affect")
    st.caption("Patient's values are compared to standard maximums (Reference ranges vary by lab).")

    # Get feature weights
    importance = get_feature_importance(model, features)
    
    # Store patient inputs in a dictionary
    patient_inputs = {
        'Age': age, 'BMI': bmi, 'Glucose': glucose, 'Insulin': insulin, 
        'BloodPressure': blood_pressure, 'Hba1c': hba1c, 'PP_Glucose': pp_glucose
    }
    
    # Create the DataFrame for the table
    # Only include features where the model weight is positive (i.e., contributing to risk)
    positive_importance = importance[importance > 0].sort_values(ascending=False)
    
    if not positive_importance.empty:
        table_data = []
        for feature, weight in positive_importance.items():
            patient_value = patient_inputs[feature]
            normal_max = NORMAL_RANGES[feature]
            
            # Determine how much the patient's value exceeds the normal max (Degree of Affect)
            degree = max(0, patient_value - normal_max)
            
            table_data.append({
                'Feature': feature,
                'Patient Value': f"{patient_value:.1f}",
                'Normal Max': f"< {normal_max:.1f}",
                'Excess Value': f"+{degree:.1f}",
                'Model Weight (Importance)': f"{weight:.2f}"
            }) # <-- THIS IS WHERE THE ERROR WAS - NOW CLOSED PROPERLY

        st.markdown("**Top Risk Drivers:**")
        # Display the data as a table/DataFrame
        st.dataframe(pd.DataFrame(table_data), hide_index=True)
    else:
        st.info("No strong positive risk drivers identified by the model.")
            
    st.info(suggestion)