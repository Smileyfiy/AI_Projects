import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="OncoGuard AI Dashboard",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CONSTANTS & CONFIG ---
DB_FILENAME = 'patient_records_db.csv'

# --- 1. HELPER FUNCTIONS FOR DATABASE (CSV) ---
def init_db(feature_cols):
    """Creates the CSV file with headers if it doesn't exist."""
    if not os.path.exists(DB_FILENAME):
        cols = ['Patient_ID'] + feature_cols
        df = pd.DataFrame(columns=cols)
        df.to_csv(DB_FILENAME, index=False)

def load_patients():
    """Loads the patient data from CSV."""
    if os.path.exists(DB_FILENAME):
        return pd.read_csv(DB_FILENAME)
    else:
        return pd.DataFrame()

def save_patient(patient_id, current_input_data):
    """Saves a new patient record to the CSV."""
    df_existing = load_patients()
    
    # Check if ID already exists
    if not df_existing.empty and patient_id in df_existing['Patient_ID'].astype(str).values:
        return False, "Patient ID already exists. Please use a unique ID."
    
    new_row = {'Patient_ID': patient_id}
    new_row.update(current_input_data)
    
    df_new = pd.DataFrame([new_row])
    # Append to existing and save
    df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    df_combined.to_csv(DB_FILENAME, index=False)
    return True, f"Patient {patient_id} saved successfully!"


# --- 2. LOAD AND PREP DATA & MODEL ---
@st.cache_data
def get_data_and_model():
    data = load_breast_cancer()
    
    # We use the first 10 'mean' features for the UI simplicity
    feature_names_raw = data.feature_names[:10]
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df = df.iloc[:, :10] 
    df['target'] = data.target
    
    # Train Model
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Scaling is crucial for accuracy
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = LogisticRegression(max_iter=3000)
    model.fit(X_scaled, y)
    
    # Get averages for benign/malignant for the radar chart comparison
    mal_avg = df[df['target']==0].drop('target', axis=1).mean()
    
    # Global Min/Max for scaling the radar chart (0-1 range)
    data_min = df.drop('target', axis=1).min()
    data_max = df.drop('target', axis=1).max()

    return df, feature_names_raw, model, scaler, mal_avg, data_min, data_max

df, feature_names_raw, model, scaler, mal_avg, data_min, data_max = get_data_and_model()

# Initialize DB with correct columns
init_db(list(feature_names_raw))

# --- 3. SIDEBAR - PATIENT MANAGEMENT ---
st.sidebar.header("Patient Management")

# Mode Selection
task_mode = st.sidebar.radio("Select Mode:", ["New Patient Entry", "Load Existing Patient"])

# Variable to hold the data that will populate the sliders
# Default to the dataset global means
slider_defaults = df.drop('target', axis=1).mean().to_dict()
current_patient_id = "New-Entry"

if task_mode == "Load Existing Patient":
    patient_db = load_patients()
    if not patient_db.empty:
        patient_list = patient_db['Patient_ID'].astype(str).tolist()
        selected_patient_id = st.sidebar.selectbox("Select Patient ID", patient_list)
        
        # Find the row for the selected patient and update slider defaults
        if selected_patient_id:
            selected_row = patient_db[patient_db['Patient_ID'].astype(str) == selected_patient_id].iloc[0]
            slider_defaults = selected_row.drop('Patient_ID').to_dict()
            current_patient_id = selected_patient_id
    else:
        st.sidebar.warning("No patients found in database.")

st.sidebar.divider()
st.sidebar.subheader(f"Clinical Data: {current_patient_id}")

# --- 4. SIDEBAR - SLIDERS GENERATION ---
# The sliders are generated dynamically.
input_data = {}
for col in feature_names_raw:
    nice_name = col.replace("mean ", "").capitalize()
    
    input_data[col] = st.sidebar.slider(
        label=nice_name,
        min_value=float(df[col].min()),
        max_value=float(df[col].max()),
        value=float(slider_defaults.get(col, df[col].mean()))
    )

# --- 5. SAVE FUNCTIONALITY (Only in New Patient Mode) ---
if task_mode == "New Patient Entry":
    st.sidebar.divider()
    new_pid_input = st.sidebar.text_input("Enter New Patient ID to Save:")
    if st.sidebar.button("💾 Save Current Patient Record"):
        if new_pid_input:
            success, msg = save_patient(new_pid_input, input_data)
            if success:
                st.sidebar.success(msg)
                # Force a rerun so the new patient appears in the list if we switch modes
                st.rerun() 
            else:
                st.sidebar.error(msg)
        else:
            st.sidebar.error("Please enter an ID.")


# --- 6. MAIN DASHBOARD ---
st.title("🩺 OncoGuard Diagnostic Dashboard")
st.markdown("""<style>.big-font { font-size:20px !important; }</style>""", unsafe_allow_html=True)
st.markdown(f'<p class="big-font">Analysis for Patient ID: <b>{current_patient_id}</b></p>', unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Biomarker Profile Visualization")
    
    # --- RADAR CHART CALCS ---
    # Scale inputs 0-1 for the chart
    input_df = pd.DataFrame([input_data])
    input_scaled_radar = (input_df.iloc[0] - data_min) / (data_max - data_min)
    mal_scaled_radar = (mal_avg - data_min) / (data_max - data_min)
    
    categories = [c.replace("mean ", "").capitalize() for c in feature_names_raw]

    fig = go.Figure()

    # Trace 1: The Malignant "Danger Zone" Profile
    fig.add_trace(go.Scatterpolar(
        r=mal_scaled_radar,
        theta=categories,
        fill='toself',
        name='Avg Malignant Profile',
        line_color='red',
        opacity=0.3,
        line=dict(dash='dashdot')  # <--- FIXED: 'dashdot' is the correct property
    ))

    # Trace 2: The Current Patient's Data
    fig.add_trace(go.Scatterpolar(
        r=input_scaled_radar,
        theta=categories,
        fill='toself',
        name='Current Patient',
        line_color='#1f77b4',
        opacity=0.8
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], gridcolor='lightgrey'),
            bgcolor='rgba(0,0,0,0)'
        ),
        showlegend=True,
        legend=dict(yanchor="top", y=1.1, xanchor="left", x=0.8),
        height=550,
        title={
            'text': "Patient vs. Malignant Benchmark",
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'}
    )
    
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("AI Diagnostic Result")
    
    # --- PREDICTION ---
    input_array = np.array(list(input_data.values())).reshape(1, -1)
    # Scale using the same scaler the model was trained on
    input_array_scaled_model = scaler.transform(input_array)
    
    prediction = model.predict(input_array_scaled_model)
    probability = model.predict_proba(input_array_scaled_model)
    
    malignant_prob = probability[0][0]
    benign_prob = probability[0][1]

    container = st.container(border=True)
    with container:
        if prediction[0] == 0:
            st.error("⚠️ MALIGNANT DETECTED")
            st.metric(label="Model Confidence", value=f"{malignant_prob*100:.2f}%")
            st.markdown("### Interpretation")
            st.write("The biomarker profile strongly aligns with patterns observed in malignant cases.")
        else:
            st.success("✅ BENIGN (SAFE)")
            st.metric(label="Model Confidence", value=f"{benign_prob*100:.2f}%")
            st.markdown("### Interpretation")
            st.write("The biomarker profile suggests benign tissue characteristics.")

    st.divider()
    st.info("**Note:** The radar chart compares relative values against the worst-case scenario. Spikes towards the outer edge (Red Zone) indicate higher risk features.")