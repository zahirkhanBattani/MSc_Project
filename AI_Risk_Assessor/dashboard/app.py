import streamlit as st
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBRegressor, Booster
import matplotlib.pyplot as plt

# ============================================================
# LOAD RESOURCES
# ============================================================

# Load input columns
INPUT_COLS = joblib.load("../notebooks/models/regressor_input_columns.pkl")

# Load booster model
booster = Booster()
booster.load_model("../notebooks/models/xgb_reg_booster_raw.json")  # or .ubj or .model
model = XGBRegressor()
model._Booster = booster

# Global thresholds
LOW_MED   = 0.4981968104839325
MED_HIGH  = 0.518235981464386
HIGH_CRIT = 0.750578984618187


# ============================================================
# PREPROCESSING & FEATURE ENGINEERING
# ============================================================

LMH_MAP = {'Low': 3, 'Medium': 5, 'High': 8}

def normalize_inputs(df):
    df = df.copy()

    lmh_cols = ['Integration_Complexity', 'Requirement_Stability', 'Market_Volatility']
    for c in lmh_cols:
        if c in df.columns:
            s = df[c].astype(str).str.strip()
            mapped = s.map(LMH_MAP)
            numeric = pd.to_numeric(s.str.replace(",", ""), errors='coerce')
            df[c] = mapped.fillna(numeric)

    numeric_cols = ['Project_Budget_USD', 'Team_Size', 'Estimated_Timeline_Months', 'Stakeholder_Count']
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", ""), errors='coerce')

    return df


def feature_engineer(df):
    df = df.copy()
    if {'Project_Budget_USD','Team_Size'}.issubset(df.columns):
        df['Budget_Per_Team'] = df['Project_Budget_USD'] / (df['Team_Size'] + 1e-5)
    if {'Estimated_Timeline_Months','Stakeholder_Count'}.issubset(df.columns):
        df['Schedule_Pressure'] = df['Estimated_Timeline_Months'] / (df['Stakeholder_Count'] + 1e-5)
    if {'Integration_Complexity','Requirement_Stability','Market_Volatility'}.issubset(df.columns):
        df['Complexity_Index'] = (
            df['Integration_Complexity'].astype(float)
            + df['Requirement_Stability'].astype(float)
            + df['Market_Volatility'].astype(float)
        ) / 3.0
    return df


def encode_and_align(df):
    df_enc = pd.get_dummies(df, drop_first=True)
    df_enc = df_enc.reindex(columns=INPUT_COLS, fill_value=0)
    return df_enc


# ============================================================
# RISK LABEL MAPPING (GLOBAL THRESHOLDS)
# ============================================================

def score_to_label(score):
    if score < LOW_MED:
        return "Low"
    elif score < MED_HIGH:
        return "Medium"
    elif score < HIGH_CRIT:
        return "High"
    else:
        return "Critical"


# ============================================================
# STREAMLIT UI
# ============================================================

st.set_page_config(page_title="AI Project Risk Assessor", layout="wide")
st.title("📊 AI-Powered Project Risk Assessment Tool")
st.write("Predict project risk using a trained XGBoost regression model and global percentile thresholds.")


tab1, tab2 = st.tabs(["🔮 Single Prediction", "📂 Batch Prediction"])


# ============================================================
# SINGLE PREDICTION TAB
# ============================================================

with tab1:

    st.header("🔮 Single Project Prediction")

    col1, col2 = st.columns(2)

    with col1:
        budget = st.number_input("Project Budget (USD)", min_value=0.0, value=50000.0)
        team = st.number_input("Team Size", min_value=1, value=5)
        timeline = st.number_input("Estimated Timeline (Months)", min_value=1, value=6)

    with col2:
        stakeholders = st.number_input("Stakeholder Count", min_value=1, value=3)
        integration = st.selectbox("Integration Complexity", ["Low", "Medium", "High"])
        stability = st.selectbox("Requirement Stability", ["Low", "Medium", "High"])
        volatility = st.selectbox("Market Volatility", ["Low", "Medium", "High"])

    if st.button("Predict Risk"):

        df_input = pd.DataFrame([{
            "Project_Budget_USD": budget,
            "Team_Size": team,
            "Estimated_Timeline_Months": timeline,
            "Stakeholder_Count": stakeholders,
            "Integration_Complexity": integration,
            "Requirement_Stability": stability,
            "Market_Volatility": volatility,
        }])

        df_input = normalize_inputs(df_input)
        df_input = feature_engineer(df_input)
        X_input = encode_and_align(df_input)

        score = model.predict(X_input)[0]
        label = score_to_label(score)

        st.success(f"Predicted Risk Level: **{label}**")
        st.write(f"Raw Score: `{score:.4f}`")


# ============================================================
# BATCH PREDICTION TAB
# ============================================================

with tab2:

    st.header("📂 Batch Risk Prediction")
    uploaded = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded:

        df = pd.read_csv(uploaded)
        st.write("### Preview of Uploaded Data")
        st.dataframe(df.head())

        df_clean = normalize_inputs(df)
        df_clean = feature_engineer(df_clean)

        X_enc = encode_and_align(df_clean)

        df["Predicted_Score"] = model.predict(X_enc)
        df["Risk_Label"] = df["Predicted_Score"].apply(score_to_label)

        st.write("### Results")
        st.dataframe(df.head())

        

        st.subheader("📊 Risk Score Distribution")

        # Create histogram
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(df["Predicted_Score"], bins=20, alpha=0.7, color="#4C72B0")

        # Add threshold lines
        ax.axvline(LOW_MED, color="green", linestyle="--", linewidth=2, label=f"Low→Medium ({LOW_MED:.3f})")
        ax.axvline(MED_HIGH, color="orange", linestyle="--", linewidth=2, label=f"Medium→High ({MED_HIGH:.3f})")
        ax.axvline(HIGH_CRIT, color="red", linestyle="--", linewidth=2, label=f"High→Critical ({HIGH_CRIT:.3f})")

        # Labels
        ax.set_xlabel("Predicted Risk Score")
        ax.set_ylabel("Frequency")
        ax.set_title("Risk Score Distribution with Thresholds")
        ax.legend()

        st.pyplot(fig)


        # Download
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download Predictions", csv, "predictions.csv", "text/csv")
























# import streamlit as st
# import pandas as pd
# import joblib
# import csv
# import numpy as np

# # =====================================================
# # 1) SAFE FEATURE ENGINEERING (bulletproof)
# # =====================================================
# def feature_engineer(df):
#     df = df.copy()

#     # Columns that must be numeric
#     num_cols = [
#         'Project_Budget_USD', 'Team_Size', 'Estimated_Timeline_Months',
#         'Stakeholder_Count', 'Integration_Complexity',
#         'Requirement_Stability', 'Market_Volatility'
#     ]

#     # Force numeric safely
#     for c in num_cols:
#         if c in df.columns:
#             df[c] = (
#                 df[c].astype(str)
#                 .str.replace(",", "", regex=True)
#                 .str.replace(" ", "", regex=True)
#             )
#             df[c] = pd.to_numeric(df[c], errors='coerce')

#     # Safe engineered features
#     if {'Project_Budget_USD','Team_Size'}.issubset(df.columns):
#         df['Budget_Per_TeamMember'] = df['Project_Budget_USD'] / (df['Team_Size'] + 1e-5)

#     if {'Estimated_Timeline_Months','Stakeholder_Count'}.issubset(df.columns):
#         df['Schedule_Pressure_Index'] = df['Estimated_Timeline_Months'] / (df['Stakeholder_Count'] + 1e-5)

#     if {'Integration_Complexity','Requirement_Stability','Market_Volatility'}.issubset(df.columns):
#         df['Complexity_Index'] = (
#             df['Integration_Complexity'] +
#             df['Requirement_Stability'] +
#             df['Market_Volatility']
#         ) / 3

#     return df


# def convert_all_columns_to_numeric(df):
#     df = df.copy()

#     for col in df.columns:
#         df[col] = pd.to_numeric(df[col], errors='coerce')
#         df[col] = df[col].fillna(0)
#         df[col] = df[col].astype(float)

#     return df


# # =====================================================
# # 2) Load model + training column schema
# # =====================================================
# pipe_reg = joblib.load('../notebooks/models/xgboost_regressor.pkl')
# reg_cols = joblib.load('../notebooks/models/regressor_input_columns.pkl')

# # Helper: Convert Risk-Score → Risk-Level
# def score_to_label(score, thresholds):
#     if score <= thresholds["Low"]:
#         return "Low"
#     elif score <= thresholds["Medium"]:
#         return "Medium"
#     elif score <= thresholds["High"]:
#         return "High"
#     return "Critical"


# def compute_thresholds_from_reference_data():
#     """
#     Compute global dynamic thresholds for the entire application.
#     Ensures single + batch prediction use SAME threshold logic,
#     while handling missing one-hot columns safely.
#     """
#     # 1) Load raw dataset
#     df = pd.read_csv("../data/project_risk_raw_dataset.csv")

#     # 2) Apply the SAME feature engineering as during training
#     df = feature_engineer(df)

#     # 3) Convert all columns to numeric (our standard helper)
#     df = convert_all_columns_to_numeric(df)

#     # 4) 🔥 Critical: add any missing one-hot columns as 0
#     #    (because reg_cols contains all encoded feature names)
#     for col in reg_cols:
#         if col not in df.columns:
#             df[col] = 0.0  # float to match model expectations

#     # 5) Reorder columns to match training order
#     df = df[reg_cols]

#     # 6) Predict scores using the trained regressor
#     scores = pipe_reg.predict(df)

#     # 7) Compute quantile-based thresholds
#     df_temp = pd.DataFrame({"Predicted_Risk_Score": scores})

#     return {
#         "Low":    df_temp["Predicted_Risk_Score"].quantile(0.25),
#         "Medium": df_temp["Predicted_Risk_Score"].quantile(0.50),
#         "High":   df_temp["Predicted_Risk_Score"].quantile(0.75),
#     }



# # Compute thresholds ONCE on app startup
# GLOBAL_THRESHOLDS = compute_thresholds_from_reference_data()


# # =====================================================
# # 3) STREAMLIT UI
# # =====================================================
# st.set_page_config(page_title="AI Risk Assessor (Regression)", layout="centered")
# st.title("📊 AI-Powered Project Risk Assessor (Regression Model)")
# st.write("Risk levels are assigned using *dynamic thresholds* learned from the full dataset.")


# # =====================================================
# # 4) SINGLE PROJECT PREDICTION
# # =====================================================
# st.header("🔹 Single Project Prediction")

# budget = st.number_input("Project Budget (USD)", 1000, 10000000, step=1000)
# team_size = st.number_input("Team Size", 1, 100)
# timeline = st.number_input("Estimated Timeline (months)", 1, 48)
# stakeholders = st.number_input("Stakeholder Count", 1, 50)
# complexity = st.slider("Integration Complexity (1-10)", 1, 10, 5)
# req_stability = st.slider("Requirement Stability (1-10)", 1, 10, 5)
# volatility = st.slider("Market Volatility (1-10)", 1, 10, 5)

# row = {col: 0 for col in reg_cols}
# row.update({
#     'Project_Budget_USD': budget,
#     'Team_Size': team_size,
#     'Estimated_Timeline_Months': timeline,
#     'Stakeholder_Count': stakeholders,
#     'Integration_Complexity': complexity,
#     'Requirement_Stability': req_stability,
#     'Market_Volatility': volatility
# })

# input_df = pd.DataFrame([row])
# input_df = feature_engineer(input_df)
# input_df = convert_all_columns_to_numeric(input_df)


# if st.button("Predict Risk Level"):
#     score = float(pipe_reg.predict(input_df)[0])
#     label = score_to_label(score, GLOBAL_THRESHOLDS)

#     st.write(f"📊 **Raw Risk Score:** {score:.3f}")
#     st.success(f"🧠 **Predicted Project Risk Level:** {label}")


# # =====================================================
# # 5) BATCH CSV PREDICTION
# # =====================================================
# st.markdown("---")
# st.header("📁 Batch Prediction (Upload CSV)")

# uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

# if uploaded_file is not None:
#     try:
#         # detect delimiter
#         sample = uploaded_file.read(2048).decode('utf-8', errors='ignore')
#         uploaded_file.seek(0)
#         delimiter = csv.Sniffer().sniff(sample).delimiter

#         df_in = pd.read_csv(uploaded_file, sep=delimiter)
#         df_in.columns = df_in.columns.str.strip().str.replace(" ", "_")

#         st.write("### ✔ Raw data preview:")
#         st.dataframe(df_in.head())

#         df_fe = feature_engineer(df_in)

#         for col in reg_cols:
#             if col not in df_fe.columns:
#                 df_fe[col] = 0

#         df_fe = df_fe[reg_cols]
#         df_fe = convert_all_columns_to_numeric(df_fe)

#         scores = pipe_reg.predict(df_fe)
#         df_in["Predicted_Risk_Score"] = scores

#         df_in["Predicted_Risk_Level"] = df_in["Predicted_Risk_Score"].apply(
#             lambda s: score_to_label(s, GLOBAL_THRESHOLDS)
#         )

#         st.success("🎯 Batch predictions completed!")
#         st.dataframe(df_in.head(20))

#         # download button
#         csv_out = df_in.to_csv(index=False).encode("utf-8")
#         st.download_button("⬇ Download Results CSV", csv_out, "risk_predictions.csv")

#     except Exception as e:
#         st.error("❌ Error during batch prediction")
#         st.code(str(e))












#################################################################################################################################
# import streamlit as st
# import pandas as pd
# import joblib
# import csv
# import traceback

# # ==========================================================
# # 🧩 1. Feature Engineering Function (same as training)
# # ==========================================================
# def feature_engineer(df):
#     df = df.copy()
#     if {'Project_Budget_USD', 'Team_Size'}.issubset(df.columns):
#         df['Budget_Per_TeamMember'] = df['Project_Budget_USD'] / (df['Team_Size'] + 1e-5)
#     if {'Estimated_Timeline_Months', 'Stakeholder_Count'}.issubset(df.columns):
#         df['Schedule_Pressure_Index'] = df['Estimated_Timeline_Months'] / (df['Stakeholder_Count'] + 1e-5)
#     if {'Integration_Complexity', 'Requirement_Stability', 'Market_Volatility'}.issubset(df.columns):
#         df['Complexity_Index'] = (
#             df['Integration_Complexity'] + df['Requirement_Stability'] + df['Market_Volatility']
#         ) / 3
#     return df


# # ==========================================================
# # 🧮 2. Categorical Encoding Mappings
# # ==========================================================
# CATEGORICAL_MAPPINGS = {
#     'Project_Type': {'Low': 0, 'Medium': 1, 'High': 2},
#     'Team_Size': {'Low': 0, 'Medium': 1, 'High': 2},
#     'Methodology_Used': {'Low': 0, 'Medium': 1, 'High': 2},
#     'Team_Experience_Level': {'Low': 0, 'Medium': 1, 'High': 2},
#     'Past_Similar_Projects': {'Low': 0, 'Medium': 1, 'High': 2},
#     'Change_Request_Frequency': {'Low': 0, 'Medium': 1, 'High': 2},
#     'Project_Phase': {'Low': 0, 'Medium': 1, 'High': 2},
#     'Requirement_Stability': {'Low': 0, 'Medium': 1, 'High': 2}
# }

# def encode_categorical_columns(df):
#     """Convert text categories to numeric codes."""
#     df = df.copy()
#     for col, mapping in CATEGORICAL_MAPPINGS.items():
#         if col in df.columns:
#             df[col] = pd.to_numeric(df[col], errors='ignore')
#             if df[col].dtype == 'object':
#                 df[col] = df[col].map(mapping)
#     return df


# # ==========================================================
# # 🎯 3. Load Full Model Pipeline and Base Columns
# # ==========================================================
# pipe = joblib.load('../notebooks/models/final_pipeline.pkl')
# base_cols = joblib.load('../notebooks/models/base_input_columns.pkl')

# # ==========================================================
# # 🖥️ 4. Streamlit UI Setup
# # ==========================================================
# st.set_page_config(page_title="AI Risk Assessor", layout="centered")
# st.title("📊 AI-Powered Project Risk Assessor")

# st.markdown("Enter key project parameters below to predict the **overall project risk level**.")

# # ==========================================================
# # 🔹 Single Record Prediction
# # ==========================================================
# budget = st.number_input("Project Budget (USD)", 1000, 10000000, step=1000)
# team_size = st.number_input("Team Size", 1, 100)
# timeline = st.number_input("Estimated Timeline (months)", 1, 48)
# stakeholders = st.number_input("Stakeholder Count", 1, 50)
# complexity = st.slider("Integration Complexity (1-10)", 1, 10, 5)
# req_stability = st.slider("Requirement Stability (1-10)", 1, 10, 5)
# volatility = st.slider("Market Volatility (1-10)", 1, 10, 5)

# # Create single-row dataframe
# row = {col: None for col in base_cols}
# row.update({
#     'Project_Budget_USD': budget,
#     'Team_Size': team_size,
#     'Estimated_Timeline_Months': timeline,
#     'Stakeholder_Count': stakeholders,
#     'Integration_Complexity': complexity,
#     'Requirement_Stability': req_stability,
#     'Market_Volatility': volatility
# })
# input_df = pd.DataFrame([row])

# if st.button("Predict Risk Level"):
#     try:
#         pred = pipe.predict(input_df)[0]
#         risk_map = {0: "Low", 1: "Medium", 2: "High", 3: "Critical"}
#         st.success(f"🧠 Predicted Project Risk Level: **{risk_map.get(pred, 'Unknown')}**")
#     except Exception as e:
#         st.error(f"❌ Prediction failed: {e}")
#         st.code(traceback.format_exc())

# # ==========================================================
# # 📂 5. Batch Prediction (CSV Upload)
# # ==========================================================
# st.markdown("---")
# st.subheader("📁 Batch Prediction (Upload CSV)")
# st.write(
#     "Upload a CSV file containing multiple project records to predict the risk level for each one."
# )

# uploaded_file = st.file_uploader(
#     "Choose a CSV file", type=["csv"], help="Ensure column names match the dataset schema."
# )

# if uploaded_file is not None:
#     try:
#         # Auto-detect delimiter
#         sniff = csv.Sniffer()
#         sample = uploaded_file.read(2048).decode('utf-8', errors='ignore')
#         uploaded_file.seek(0)
#         delimiter = sniff.sniff(sample).delimiter

#         df_in = pd.read_csv(uploaded_file, sep=delimiter, engine='python')

#         # Clean column names
#         df_in.columns = df_in.columns.str.strip().str.replace('\ufeff', '', regex=True).str.replace(' ', '_')

#         st.write("✅ File uploaded successfully!")
#         st.write("**Original Data Preview:**")
#         st.dataframe(df_in.head())

#         # Encode categorical values
#         df_in = encode_categorical_columns(df_in)

#         # Convert numeric-looking values
#         for col in df_in.columns:
#             df_in[col] = pd.to_numeric(df_in[col], errors='ignore')

#         # 🔧 Apply feature engineering
#         df_in = feature_engineer(df_in)

#         # Align columns to base schema
#         missing_cols = [c for c in base_cols if c not in df_in.columns]
#         for col in missing_cols:
#             df_in[col] = None
#         df_in = df_in[base_cols]

#         # Make predictions
#         preds = pipe.predict(df_in)

#         # Map predictions to labels
#         risk_map = {0: "Low", 1: "Medium", 2: "High", 3: "Critical"}
#         df_in["Predicted_Risk_Level"] = [risk_map.get(p, "Unknown") for p in preds]

#         # Show output
#         st.success("🎯 Predictions completed!")
#         st.dataframe(df_in[['Predicted_Risk_Level']].head(20))

#         # Download predictions
#         csv_out = df_in.to_csv(index=False).encode("utf-8")
#         st.download_button(
#             label="⬇️ Download Predictions as CSV",
#             data=csv_out,
#             file_name="risk_predictions.csv",
#             mime="text/csv",
#         )

#     except Exception as e:
#         st.error(f"❌ Batch prediction failed: {e}")
#         st.write("**Debug Info:**")
#         st.code(traceback.format_exc())
