import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ================== PAGE CONFIG ==================
st.set_page_config(page_title="Event Attendance Prediction", layout="wide")

st.title("🎟️ Event Attendance Prediction System")

# ================== LOAD MODELS ==================
model = joblib.load("models/linear_model.pkl")
rf_model = joblib.load("models/rf_model.pkl")
ohe = joblib.load("models/ohe_encoder.pkl")
mms = joblib.load("models/scaler.pkl")

# ================== INPUT FORM ==================
st.header("Enter Event Details")

col1, col2 = st.columns(2)

with col1:
    event_type = st.selectbox("Event Type", ["sports", "concert", "festival", "traditional events", "pre-release"])
    sub_event_type = st.text_input("Sub Event Type", "cricket")
    venue_city = st.selectbox("Venue City", ["chennai", "hyderabad", "vijayawada", "bengaluru", "delhi"])
    season = st.selectbox("Season", ["summer", "winter", "autumn", "rainy"])
    weather = st.selectbox("Weather", ["sunny", "cloudy", "rainy", "windy"])

with col2:
    ticket_price = st.number_input("Ticket Price", min_value=0, max_value=5000, value=1200)
    avg_past_attendance = st.number_input("Avg Past Attendance", min_value=0, max_value=60000, value=20000)
    venue_capacity = st.number_input("Venue Capacity", min_value=1000, max_value=100000, value=50000)
    weekday = st.selectbox("Weekday", ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"])
    month = st.selectbox("Month", [
        "january","february","march","april","may","june","july",
        "august","september","october","november","december"
    ])

# Holiday flags
is_holiday = 1 if weekday in ["saturday", "sunday"] else 0
is_weekend = is_holiday

# ================== DF CONSTRUCTION ==================
input_df = pd.DataFrame({
    "event_type":[event_type.lower()],
    "sub_event_type":[sub_event_type.lower()],
    "venue_city":[venue_city.lower()],
    "season":[season.lower()],
    "weather":[weather.lower()],
    "ticket_price":[ticket_price],
    "avg_past_attendance":[avg_past_attendance],
    "venue_capacity":[venue_capacity],
    "weekday":[weekday.lower()],
    "month":[month.lower()],
    "is_holiday":[is_holiday],
    "is_weekend":[is_weekend],
})

st.write("### ✅ Clean Input Data", input_df)

# ================== PREPROCESS ==================
# --- One Hot Encoding ---
ohe_cols = ["event_type","sub_event_type","venue_city","season","weather","weekday","month"]
ohe_data = ohe.transform(input_df[ohe_cols]).toarray()
ohe_df = pd.DataFrame(ohe_data, columns=ohe.get_feature_names_out())

# --- Scaling ---
scale_cols = ["venue_capacity","ticket_price","is_holiday","is_weekend","avg_past_attendance"]
scaled = mms.transform(input_df[scale_cols])
scaled_df = pd.DataFrame(scaled, columns=scale_cols)

# --- Final X ---
X_input = pd.concat([scaled_df, ohe_df], axis=1)

# ================== PREDICT ==================
if st.button("Predict Attendance"):
    linear_pred = model.predict(X_input)[0]
    rf_pred = rf_model.predict(X_input)[0]

    st.success(f"📌 Linear Regression Prediction: {int(linear_pred)} people")
    st.success(f"🌲 Random Forest Prediction: {int(rf_pred)} people")
