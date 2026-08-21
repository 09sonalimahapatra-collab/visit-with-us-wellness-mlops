"""Streamlit frontend for Wellness Tourism Package purchase prediction."""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "wellness_package_model.joblib"

st.set_page_config(page_title="Wellness Package Predictor", page_icon="✈️", layout="centered")


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model not found. Run the GitHub Actions pipeline successfully before deploying this app."
        )
    return joblib.load(MODEL_PATH)


st.title("Wellness Tourism Package Predictor")
st.caption("Use the customer's profile and interaction details to prioritize outreach.")

with st.form("customer_profile"):
    st.subheader("Customer profile")
    left, right = st.columns(2)
    with left:
        age = st.number_input("Age", min_value=18, max_value=100, value=35, step=1)
        contact = st.selectbox("Type of contact", ["Self Enquiry", "Company Invited"])
        city_tier = st.selectbox("City tier", [1, 2, 3], index=1)
        occupation = st.selectbox(
            "Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"]
        )
        gender = st.selectbox("Gender", ["Female", "Male"])
        marital_status = st.selectbox("Marital status", ["Single", "Married", "Divorced"])
        monthly_income = st.number_input("Monthly income", min_value=0, value=25000, step=500)
    with right:
        people_visiting = st.number_input("Number of people visiting", min_value=1, value=2, step=1)
        followups = st.number_input("Number of follow-ups", min_value=0, value=3, step=1)
        duration = st.number_input("Pitch duration (minutes)", min_value=0.0, value=12.0, step=1.0)
        product = st.selectbox("Product pitched", ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"])
        star_rating = st.selectbox("Preferred property star rating", [3, 4, 5], index=1)
        trips = st.number_input("Average annual trips", min_value=0, value=2, step=1)
        designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])

    st.subheader("Travel indicators")
    first, second, third = st.columns(3)
    with first:
        passport = st.selectbox("Has passport", [0, 1], format_func=lambda value: "Yes" if value else "No")
    with second:
        own_car = st.selectbox("Owns a car", [0, 1], format_func=lambda value: "Yes" if value else "No")
    with third:
        children = st.number_input("Children visiting", min_value=0, value=0, step=1)
    pitch_satisfaction = st.slider("Pitch satisfaction score", min_value=1, max_value=5, value=3)
    submitted = st.form_submit_button("Predict purchase likelihood", type="primary")

if submitted:
    input_frame = pd.DataFrame(
        [
            {
                "Age": float(age),
                "TypeofContact": contact,
                "CityTier": int(city_tier),
                "DurationOfPitch": float(duration),
                "Occupation": occupation,
                "Gender": gender,
                "NumberOfPersonVisiting": int(people_visiting),
                "NumberOfFollowups": float(followups),
                "ProductPitched": product,
                "PreferredPropertyStar": float(star_rating),
                "MaritalStatus": marital_status,
                "NumberOfTrips": float(trips),
                "Passport": int(passport),
                "PitchSatisfactionScore": int(pitch_satisfaction),
                "OwnCar": int(own_car),
                "NumberOfChildrenVisiting": float(children),
                "Designation": designation,
                "MonthlyIncome": float(monthly_income),
            }
        ]
    )
    model = load_model()
    probability = float(model.predict_proba(input_frame)[:, 1][0])
    prediction = int(probability >= 0.50)

    st.subheader("Prediction")
    st.metric("Purchase probability", f"{probability:.1%}")
    if prediction:
        st.success("High-priority lead: consider a personalized follow-up for the Wellness Tourism Package.")
    else:
        st.info("Lower-priority lead at the current threshold: use a lower-cost or nurture campaign first.")

    with st.expander("View dataframe sent to the model"):
        st.dataframe(input_frame, use_container_width=True)

st.divider()
st.caption("Decision support only: the marketing team should combine this score with consent and campaign rules.")
