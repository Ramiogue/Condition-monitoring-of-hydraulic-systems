import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- Load the trained model ---
model = joblib.load("model.pkl")

# --- Interpret model output into human-friendly messages ---
def interpret_predictions(preds):
    messages = []

    # Cooler
    cooler = preds[0]
    if cooler == 100:
        messages.append("✅ The cooler is functioning normally.")
    elif cooler == 20:
        messages.append("🟡 The cooler is degraded – consider cleaning or inspecting.")
    elif cooler == 3:
        messages.append("🔴 The cooler is near failure – immediate maintenance required.")

    # Valve
    valve = preds[1]
    if valve == 100:
        messages.append("✅ The valve is switching normally.")
    elif valve == 90:
        messages.append("🟡 The valve is slightly delayed – monitor it.")
    elif valve == 80:
        messages.append("🟠 The valve is sluggish – schedule inspection.")
    elif valve == 73:
        messages.append("🔴 The valve is failing – needs urgent attention.")

    # Pump leakage
    pump = preds[2]
    if pump == 0:
        messages.append("✅ The pump is sealed properly.")
    elif pump == 1:
        messages.append("🟡 The pump shows weak leakage – schedule maintenance.")
    elif pump == 2:
        messages.append("🔴 Severe leakage detected – fix ASAP.")

    # Accumulator pressure
    acc = preds[3]
    if acc == 130:
        messages.append("✅ Accumulator pressure is optimal.")
    elif acc == 115:
        messages.append("🟡 Slight pressure drop – check soon.")
    elif acc == 100:
        messages.append("🟠 Low pressure – re-pressurize soon.")
    elif acc == 90:
        messages.append("🔴 Critical low pressure – immediate action required.")

    return messages

# --- Feature extraction from a raw sensor file ---
def extract_features(df):
    features = {}
    for col in df.columns:
        signal = df[col]
        features[f"{col}_mean"] = signal.mean()
        features[f"{col}_std"] = signal.std()
        features[f"{col}_min"] = signal.min()
        features[f"{col}_max"] = signal.max()
        features[f"{col}_median"] = signal.median()
        features[f"{col}_ptp"] = np.ptp(signal)
        features[f"{col}_var"] = signal.var()
        features[f"{col}_skew"] = signal.skew()
        features[f"{col}_kurtosis"] = signal.kurtosis()
        features[f"{col}_first_min"] = signal.idxmin()
        features[f"{col}_first_max"] = signal.idxmax()
        features[f"{col}_last_min"] = signal[::-1].idxmin()
        
        features[f"{col}_sum"] = signal.sum()
        features[f"{col}_abs_energy"] = np.sum(signal ** 2)
        features[f"{col}_abs_sum_change"] = np.sum(np.abs(np.diff(signal)))
        features[f"{col}_mean_abs_change"] = np.mean(np.abs(np.diff(signal)))
    return pd.DataFrame([features])

# --- Streamlit App UI ---
st.set_page_config(page_title="Hydraulic Condition Monitor", page_icon="🛠")
st.title("🛠 Hydraulic Condition Monitoring")
st.markdown("Upload a **raw sensor file** (e.g. from PS1, TS2, etc.) to predict the current equipment conditions.")

uploaded_file = st.file_uploader("📂 Upload Sensor File (.txt or .csv)", type=["txt", "csv"])

if uploaded_file is not None:
    try:
        # Try reading with tab or comma
        try:
            df = pd.read_csv(uploaded_file, sep="\t", header=None)
        except:
            df = pd.read_csv(uploaded_file, sep=",", header=None)

        st.success("✅ File uploaded successfully!")
        st.subheader("📊 Preview of Raw Sensor Data:")
        st.dataframe(df.head())

        # Feature extraction
        with st.spinner("⚙️ Extracting features..."):
            X = extract_features(df)

        # Prediction
        with st.spinner("🧠 Running model prediction..."):
            preds = model.predict(X)[0]

        st.subheader("🧠 Predicted Condition Codes:")
        st.write({
            "Cooler condition": preds[0],
            "Valve condition": preds[1],
            "Pump leakage": preds[2],
            "Accumulator pressure": preds[3]
        })

        # Friendly messages
        st.subheader("🔎 Maintenance Suggestions:")
        for msg in interpret_predictions(preds):
            st.markdown(f"- {msg}")

    except Exception as e:
        st.error(f"⚠️ Error reading file or making prediction: {e}")
else:
    st.info("👈 Please upload a raw sensor file to begin.")
