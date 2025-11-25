
import streamlit as st
st.set_page_config(page_title="Liver & Cancer Prediction", layout="wide")
import numpy as np
import pandas as pd
import joblib



# -------------------------------
# Page Config
# -------------------------------


# -------------------------------
# Custom CSS
# -------------------------------
st.markdown(
    """
    <style>
    body {
        background-color: #f0f8ff;
    }
    .main {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 15px;
    }
    h1, h2, h3 {
        color: #004080;
    }
    .stButton button {
        background: linear-gradient(90deg, #28a745, #20c997);
        color: white;
        font-weight: bold;
        border-radius: 12px;
        padding: 12px 24px;
        border: none;
    }
    .stButton button:hover {
        background: linear-gradient(90deg, #218838, #17a2b8);
        color: white;
    }
    .input-card {
        background-color: #e6f7ff;
        border: 2px solid #99ccff;
        padding: 15px;
        border-radius: 12px;
        margin-bottom: 20px;
    }
    .prediction-box {
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        font-size: 22px;
        font-weight: bold;
        color: white;
        margin-top: 30px;
    }
    .healthy {
        background-color: #28a745;
        border: 3px solid #1e7e34;
    }
    .disease {
        background-color: #dc3545;
        border: 3px solid #a71d2a;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------------
# Sidebar Navigation
# -------------------------------
st.sidebar.title("🩺 Navigation")
page = st.sidebar.radio("Go to:", ["Home", "Predict"])

# -------------------------------
# Home Page
# -------------------------------
if page == "Home":
    st.title("🩺 Liver & Cancer Prediction System")
    st.markdown(
        """
        Welcome to the **Liver & Cancer Prediction System**.  

        ---
        > *“The liver is a resilient organ – treat it with care.”*  
        > *“An ounce of prevention is worth a pound of cure.”*  
        ---
        """
    )

    # Add images (place yours in static/images/)
    col1, col2 = st.columns(2)
    with col1:
        st.image("static/images/liver1.jpg", caption="Human Liver")
    with col2:
        st.image("static/images/liver2.jpg", caption="Liver Position in Human Body")

# -------------------------------
# KEEP YOUR EXISTING WORKING CODE BELOW
# -------------------------------


# -------------------------------
# Load Binary Model (compressed)
# -------------------------------
old_model_data = joblib.load("liver_model_compressed.pkl")
model = old_model_data["model"]
scaler = old_model_data["scaler"]
imputer = old_model_data["imputer"]

# -------------------------------
# Load Cluster Model (Disease Classification)
# -------------------------------
cluster_data = joblib.load("liver_cluster_model_remapped_greedy.pkl")
cluster_model = cluster_data["cluster_model"]
imputer_cluster = cluster_data["imputer"]
scaler_cluster = cluster_data["scaler"]
disease_map = cluster_data["disease_map"]

# Cancer mapping
cancer_map = {
    0: "Cancer",
    1: "No Cancer",
    2: "Cancer",
    3: "Cancer",
    4: "Cancer"
}

# -------------------------------
# Confidence Helper
# -------------------------------
def confidence_label(prob):
    if prob < 0.4:
        return "Very Low"
    elif prob < 0.6:
        return "Low"
    elif prob < 0.8:
        return "High"
    else:
        return "Very High"

# -------------------------------
# Prediction Function
# -------------------------------
def predict_case(input_df, cancer_check=True):
    # Normalize to correct spelling
    if "Total_Proteins" in input_df.columns:
        input_df.rename(columns={"Total_Proteins": "Total_Proteins"}, inplace=True)

    # -------- Binary model expects 'Total_Proteins' --------
    input_binary = input_df.rename(columns={"Total_Proteins": "Total_Proteins"})
    X_imp = imputer.transform(input_binary)
    X_scaled = scaler.transform(X_imp)
    pred = model.predict(X_scaled)[0]
    prob = model.predict_proba(X_scaled)[0][pred]
    conf = confidence_label(prob)

    if pred == 0:
        return {"Disease": "No Disease", "Confidence": conf}

    # -------- Cluster model expects 'Total_Proteins' --------
    input_cluster = input_df.rename(columns={"Total_Proteins": "Total_Proteins"})
    Xc_imp = imputer_cluster.transform(input_cluster)
    Xc_scaled = scaler_cluster.transform(Xc_imp)
    cluster_pred = cluster_model.predict(Xc_scaled)[0]
    disease_type = disease_map.get(cluster_pred, "Unknown")

    result = {"Disease": disease_type, "Confidence": conf}
    if cancer_check:
        result["Cancer_Risk"] = cancer_map.get(cluster_pred, "Unknown")

    return result

# -------------------------------
# Page Config
# -------------------------------




# -------------------------------
# Home Page
# -------------------------------
if page == "Home":
    st.title("🩺 Liver & Cancer Prediction System")
    st.markdown(
        """
        Welcome to the **Liver & Cancer Prediction System**.  

        ---
        > *“The liver is a resilient organ – treat it with care.”*  
        > *“An ounce of prevention is worth a pound of cure.”*  
        ---
        """
    )


# -------------------------------
# Prediction Page
# -------------------------------
elif page == "Predict":
    st.header("🔮 Predict Liver Disease & Cancer")
    st.markdown("Choose between **manual input** or **CSV upload**:")

    mode = st.radio("Select Mode:", ["Single Input", "Batch CSV Upload"])
    cancer_check = st.checkbox("Also check for Cancer if disease detected", value=True)

    # -------------------------------
    # Single Input
    # -------------------------------
    if mode == "Single Input":
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)

            with col1:
                age = st.number_input("Age", 1, 120, 30)
                gender = st.radio("Gender", ["Male", "Female"])
                gender = 1 if gender == "Male" else 0
                total_bilirubin = st.number_input("Total Bilirubin", 0.0, 50.0, 1.0)
                direct_bilirubin = st.number_input("Direct Bilirubin", 0.0, 25.0, 0.2)

            with col2:
                alk_phos = st.number_input("Alkaline Phosphotase", 10, 2000, 200)
                sgpt = st.number_input("Alamine Aminotransferase", 0, 2000, 30)
                sgot = st.number_input("Aspartate Aminotransferase", 0, 2000, 35)
                total_protein = st.number_input("Total Proteins", 0.0, 10.0, 6.5)
                albumin = st.number_input("Albumin", 0.0, 6.0, 3.5)
                ag_ratio = st.number_input("Albumin and Globulin Ratio", 0.0, 3.0, 1.0)

            submitted = st.form_submit_button("🔍 Predict Now")

        if submitted:
            input_df = pd.DataFrame([[
                age, gender, total_bilirubin, direct_bilirubin,
                alk_phos, sgpt, sgot, total_protein, albumin, ag_ratio
            ]], columns=[
                "Age", "Gender", "Total_Bilirubin", "Direct_Bilirubin",
                "Alkaline_Phosphotase", "Alamine_Aminotransferase",
                "Aspartate_Aminotransferase", "Total_Proteins", "Albumin",
                "Albumin_and_Globulin_Ratio"
            ])

            result = predict_case(input_df, cancer_check=cancer_check)

            if result["Disease"] == "No Disease":
                st.success(f"✅ No Liver Disease Detected (Confidence: {result['Confidence']})")
            else:
                st.error(f"⚠️ {result['Disease']} Detected (Confidence: {result['Confidence']})")
                

    # -------------------------------
    # CSV Upload
    # -------------------------------
    elif mode == "Batch CSV Upload":
        uploaded_file = st.file_uploader("Upload CSV File with Test Cases", type=["csv"])
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            if "Gender" in df.columns:
                df["Gender"] = df["Gender"].astype(str).str.strip().str.lower()
                df["Gender"] = df["Gender"].replace({
                    "male": 0, "m": 0,
                    "female": 1, "f": 1
                })
            else:
            	st.warning("⚠️ Gender column missing in CSV!")
            
            
            # Normalize once
            df.rename(columns={"Total_Proteins": "Total_Proteins"}, inplace=True)

            predictions = []
            for _, row in df.iterrows():
                row_df = pd.DataFrame([row])
                predictions.append(predict_case(row_df, cancer_check=cancer_check))

            # Flatten predictions
            df["Disease"] = [p["Disease"] for p in predictions]
            df["Confidence"] = [p["Confidence"] for p in predictions]
            
             

            st.subheader("📊 Batch Predictions")
            st.dataframe(df)

            # Download option
            csv_out = df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download Results", csv_out, "predictions.csv", "text/csv")
