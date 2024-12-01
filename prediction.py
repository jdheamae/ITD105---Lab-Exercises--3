import streamlit as st
import pandas as pd
import numpy as np
import joblib
import io

# Streamlit App Title
st.title("Heart Disease Prediction")

# Sidebar for navigation
option = st.sidebar.selectbox("Choose a mode", ["Batch Prediction", "Single Input Prediction"])

# ---- Load Model Section ----
st.sidebar.header("Upload Trained Model")
uploaded_model_file = st.sidebar.file_uploader("Upload Trained Model (.pkl)", type=["pkl"])

# Ensure the model is uploaded
if uploaded_model_file:
    try:
        model = joblib.load(uploaded_model_file)
        st.sidebar.success("Model loaded successfully!")
    except Exception as e:
        st.sidebar.error(f"Failed to load model: {e}")
        st.stop()
else:
    st.sidebar.info("Please upload your trained model to proceed.")
    st.stop()

# ---- Batch Prediction Section ----
if option == "Batch Prediction":
# Header for single input prediction
    st.header("Single Input Prediction")
    st.write("Manually input the details to get a prediction.")

    # Input fields for manual prediction
    X = st.number_input("X-axis spatial coordinate within the Montesinho park map", min_value=0, max_value=9, value=5)
    Y = st.number_input("Y-axis spatial coordinate within the Montesinho park map", min_value=0, max_value=9, value=5)
    month = st.selectbox("Month (1 = January, ..., 12 = December)", list(range(1, 13)))
    day = st.selectbox("Day of the week (1 = Monday, ..., 7 = Sunday)", list(range(1, 8)))
    FFMC = st.number_input("FFMC Index (Fine Fuel Moisture Code)", min_value=0.0, max_value=100.0, value=85.0)
    DMC = st.number_input("DMC Index (Duff Moisture Code)", min_value=0.0, max_value=200.0, value=25.0)
    DC = st.number_input("DC Index (Drought Code)", min_value=0.0, max_value=800.0, value=200.0)
    ISI = st.number_input("ISI Index (Initial Spread Index)", min_value=0.0, max_value=50.0, value=10.0)
    temp = st.number_input("Temperature (in Celsius)", min_value=-5.0, max_value=50.0, value=15.0)
    RH = st.number_input("Relative Humidity (%)", min_value=0, max_value=100, value=50)
    wind = st.number_input("Wind Speed (in km/h)", min_value=0.0, max_value=50.0, value=5.0)
    rain = st.number_input("Rain (in mm)", min_value=0.0, max_value=10.0, value=0.0)

    # Combine inputs into a NumPy array
    input_data = np.array([[X, Y, month, day, FFMC, DMC, DC, ISI, temp, RH, wind, rain]])

    # Display the input array for verification
    st.write("Input Data for Prediction:")
    st.write(pd.DataFrame(input_data, columns=[
    "X", "Y", "month", "day", "FFMC", "DMC", "DC", "ISI", "temp", "RH", "wind", "rain"
    ]))

if st.button("Predict Burned Area"):
    if 'model' in locals():  # Check if the model exists
        try:
            # Model prediction, the model expects input_data to be in the same shape as it was trained on
            prediction = model.predict(input_data)  # Prediction of burned area

            # Displaying the predicted burned area
            st.write("Prediction Result:")
            st.write(f"Predicted Burned Area: {prediction[0]:.3f} ha")
        except Exception as e:
            st.error(f"Error making prediction: {e}")

# ---- Single Input Prediction Section ----
elif option == "Single Input Prediction":
    st.header("Single Input Prediction")
    st.write("Manually input the details to get a prediction.")

    # User input fields
    age = st.number_input("Age", min_value=0, max_value=120, value=50)
    sex = st.selectbox("Sex (0: Female, 1: Male)", [0, 1])
    chest_pain_type = st.selectbox("Chest Pain Type (1: Typical Angina, 2: Atypical Angina, 3: Non-Anginal Pain, 4: Asymptomatic)", [1, 2, 3, 4])
    bp = st.number_input("Resting Blood Pressure (BP)", min_value=0, value=120)
    cholesterol = st.number_input("Cholesterol (mg/dl)", min_value=0, value=200)
    fbs_over_120 = st.selectbox("Fasting Blood Sugar > 120 mg/dl (0: No, 1: Yes)", [0, 1])
    ekg_results = st.selectbox("Resting EKG Results (0: Normal, 1: ST-T Abnormality, 2: Left Ventricular Hypertrophy)", [0, 1, 2])
    max_hr = st.number_input("Maximum Heart Rate Achieved", min_value=50, max_value=250, value=150)
    exercise_angina = st.selectbox("Exercise Induced Angina (0: No, 1: Yes)", [0, 1])
    st_depression = st.number_input("ST Depression Induced by Exercise", min_value=0.0, value=0.0, step=0.1)
    slope = st.selectbox("Slope of the Peak Exercise ST Segment (1: Upsloping, 2: Flat, 3: Downsloping)", options=[1, 2, 3])
    num_vessels_fluro = st.selectbox("Number of Major Vessels Colored by Fluoroscopy (0-3)", [0, 1, 2, 3])
    thallium = st.selectbox("Thallium Stress Test Result (3: Normal, 6: Fixed Defect, 7: Reversible Defect)", options=[3, 6, 7])

    # Prepare input data for prediction
    input_data = np.array([[
        age, sex, chest_pain_type, bp, cholesterol, fbs_over_120, 
        ekg_results, max_hr, exercise_angina, st_depression, 
        slope, num_vessels_fluro, thallium
    ]])

    # Make prediction
    if st.button("Predict"):
        try:
            prediction = model.predict(input_data)[0]

            # Display the result
            if prediction == 1:
                st.error("The model predicts the presence of heart disease.")
            else:
                st.success("The model predicts no heart disease.")
        except Exception as e:
            st.error(f"Error: {e}")
