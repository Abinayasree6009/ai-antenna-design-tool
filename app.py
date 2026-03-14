import streamlit as st
import numpy as np
import joblib
import pandas as pd
from tensorflow.keras.models import load_model

# Load model and scaler
model = load_model("s11_dnn_model.keras", compile=False)
scaler = joblib.load("input_scaler.pkl")

st.set_page_config(page_title="AI Antenna Design Tool", layout="wide")

st.title("AI-Based Triangle Slot Antenna Design Tool")

tabs = st.tabs(["Forward Prediction", "Inverse Design"])


# ---------------- FORWARD TAB ----------------
with tabs[0]:

    st.header("Forward Prediction (Geometry → S11 Curve)")

    lif2 = st.number_input("Inset Feed Length 2 (mm)", min_value=14, max_value=18, value=16)
    lif1 = st.number_input("Inset Feed Length 1 (mm)", min_value=18, max_value=21, value=19)
    lf = st.number_input("Feed Length (mm)", min_value=23, max_value=27, value=25)
    a = st.number_input("Patch Radius (mm)", min_value=47.0, max_value=50.0, value=48.5)

    if st.button("Generate S11 Curve"):

        freq_range = np.linspace(2.0, 3.0, 200)

        inputs = np.array([[lif2, lif1, lf, a, f] for f in freq_range])

        scaled = scaler.transform(inputs)

        predictions = model.predict(scaled, verbose=0).flatten()

        min_index = np.argmin(predictions)
        resonant_freq = freq_range[min_index]
        min_s11 = predictions[min_index]

        chart_data = pd.DataFrame({
            "Frequency (GHz)": freq_range,
            "S11 (dB)": predictions
        })

        st.line_chart(chart_data.set_index("Frequency (GHz)"))

        st.subheader("Resonance Information")

        st.write("Resonant Frequency:", round(resonant_freq,3), "GHz")
        st.write("Minimum S11:", round(min_s11,2), "dB")

        if min_s11 <= -40:
            st.success("Outstanding Matching (≤ -40 dB)")
        elif min_s11 <= -30:
            st.success("Excellent Matching (≤ -30 dB)")
        elif min_s11 <= -20:
            st.success("Very Good Matching (≤ -20 dB)")
        elif min_s11 <= -10:
            st.info("Acceptable Matching (≤ -10 dB)")
        else:
            st.error("Poor Matching (> -10 dB)")


# ---------------- INVERSE TAB ----------------
with tabs[1]:

    st.header("Inverse Design (Target Frequency → Best Geometry)")

    target_freq = st.slider("Target Frequency (GHz)", 2.0, 3.0, 2.4, step=0.01)

    if st.button("Find Best Designs"):

        st.write("Searching best geometries...")

        lif2_range = range(14,19)
        lif1_range = range(18,22)
        lf_range = range(23,28)
        a_range = np.linspace(47.0,50.0,20)

        combinations = np.array([
            [lif2_val,lif1_val,lf_val,a_val,target_freq]
            for lif2_val in lif2_range
            for lif1_val in lif1_range
            for lf_val in lf_range
            for a_val in a_range
        ])

        scaled = scaler.transform(combinations)

        preds = model.predict(scaled, verbose=0).flatten()

        df = pd.DataFrame(combinations, columns=[
            "Inset Feed Length 2 (mm)",
            "Inset Feed Length 1 (mm)",
            "Feed Length (mm)",
            "Patch Radius (mm)",
            "Frequency (GHz)"
        ])

        df["Predicted S11 (dB)"] = preds

        df = df.sort_values("Predicted S11 (dB)")

        st.subheader("Top 5 Strongest Matching Designs")

        top5 = df.head(5).reset_index(drop=True)

        st.dataframe(top5[[
            "Inset Feed Length 2 (mm)",
            "Inset Feed Length 1 (mm)",
            "Feed Length (mm)",
            "Patch Radius (mm)",
            "Predicted S11 (dB)"
        ]])

        csv = top5.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download Top 5 Designs as CSV",
            data=csv,
            file_name="inverse_design_results.csv",
            mime="text/csv"
        )
