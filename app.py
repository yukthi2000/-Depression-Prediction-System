import streamlit as st
import pickle
import pandas as pd
from catboost import CatBoostClassifier, Pool

# Load the CatBoost model
with open("CatBoost_model.pkl", "rb") as file:
    catboost_model = pickle.load(file)

# Dropdown options
city_options = ['Ludhiana', 'Varanasi', 'Visakhapatnam', 'Mumbai', 'Kanpur', 'Ahmedabad', 'Thane', 'Nashik', 'Bangalore',
                'Patna', 'Rajkot', 'Jaipur', 'Pune', 'Lucknow', 'Meerut', 'Agra', 'Surat', 'Faridabad', 'Hyderabad',
                'Srinagar', 'Ghaziabad', 'Kolkata', 'Chennai', 'Kalyan', 'Nagpur', 'Vadodara', 'Vasai-Virar', 'Delhi',
                'Bhopal', 'Indore', 'Ishanabad', 'Vidhi', 'Ayush', 'Gurgaon', 'Krishna', 'Aishwarya', 'Keshav', 'Harsha',
                'Nalini', 'Aditya', 'Malyansh', 'Raghavendra', 'Saanvi', 'M.Tech', 'Bhavna', 'Nandini', 'M.Com', 'Plata',
                'Atharv', 'Pratyush', 'Less than 5 Kalyan', 'MCA', 'Mira', 'Moreadhyay', 'Morena', 'Ishkarsh', 'Kashk',
                'Mihir', 'Vidya', 'Tolkata', 'Anvi', 'Krinda', 'Ayansh', 'Shrey', 'Ivaan', 'Vaanya', 'Gaurav', 'Harsh',
                'Reyansh', 'Kashish', 'Kibara', 'Vaishnavi', 'Chhavi', 'Parth', 'Mahi', 'Tushar', 'MSc', 'Rashi', 'ME',
                'Researcher', 'Kagan', 'Armaan', 'Ithal', 'Nalyan', 'Dhruv', 'Galesabad', 'Itheg', 'Aaradhya', 'Pooja',
                'Khushi', 'Khaziabad', 'Jhanvi', 'Unirar']

professional_options = ['Chef', 'Teacher', 'Business Analyst', 'Financial Analyst', 'Chemist', 'Electrician',
                     'Software Engineer', 'Data Scientist', 'Plumber', 'Marketing Manager', 'Accountant', 'Entrepreneur',
                     'HR Manager', 'UX/UI Designer', 'Content Writer', 'Architect', 'Educational Consultant',
                     'Civil Engineer', 'Manager', 'Pharmacist', 'Customer Support', 'Dev', 'Mechanical Engineer',
                     'Consultant', 'Analyst', 'Judge', 'Researcher', 'Research Analyst', 'Lawyer', 'Pilot',
                     'Graphic Designer', 'Travel Consultant', 'Digital Marketer', 'Sales Executive', 'Doctor',
                     'Working Professional', 'Medical Doctor', 'Family Consultant', 'Investment Banker', 'City Manager']

degree_options = ['BHM', 'LLB', 'B.Pharm', 'BBA', 'MCA', 'MD', 'BSc', 'ME', 'B.Arch', 'BCA', 'BE', 'MA', 'B.Ed', 'B.Com',
                  'MBA', 'M.Com', 'MHM', 'BA', 'Class 12', 'M.Tech', 'PhD', 'M.Ed', 'MSc', 'B.Tech', 'LLM', 'MBBS',
                  'M.Pharm', 'MPA']

sleep_duration_options = ['8-10', '0-4', '4-6', '6-8', '10-20']

st.title("Mental Health Prediction using CatBoost")

# Create two columns for the form
col1, col2 = st.columns(2)

with col1:
    # Basic Information
    st.subheader("Basic Information")
    gender = st.selectbox("Select Gender", ["Male", "Female"])
    role = st.selectbox("Working Professional or Student", ["Student", "Working Professional"])
    city = st.selectbox("Select City", city_options)
    
    # Conditional profession selection based on role
    if role == "Student":
        profession = "Student"
        st.info("Profession is set to 'Student'")
    else:
        profession = st.selectbox("Select Profession", professional_options)
    
    degree = st.selectbox("Select Degree", degree_options)
    sleep_duration = st.selectbox("Select Sleep Duration", sleep_duration_options)

with col2:
    # Health Information
    st.subheader("Health Information")
    age = st.slider("Age", 0.0, 100.0, 25.0, 1.0)
    
    # Pressure and Satisfaction (conditional)
    if role == "Student":
        academic_pressure = st.slider("Academic Pressure", 0.0, 5.0, 2.5, 0.1)
        study_satisfaction = st.slider("Study Satisfaction", 0.0, 5.0, 2.5, 0.1)
        work_pressure = 0.0
        job_satisfaction = 0.0
        st.info("Work Pressure and Job Satisfaction are set to 0 for students")
    else:
        work_pressure = st.slider("Work Pressure", 0.0, 5.0, 2.5, 0.1)
        job_satisfaction = st.slider("Job Satisfaction", 0.0, 5.0, 2.5, 0.1)
        academic_pressure = 0.0
        study_satisfaction = 0.0
        st.info("Academic Pressure and Study Satisfaction are set to 0 for working professionals")

    work_study_hours = st.slider("Work/Study Hours", 0.0, 24.0, 8.0, 0.5)
    financial_stress = st.slider("Financial Stress", 0.0, 5.0, 2.5, 0.1)

# Medical History
st.subheader("Medical History")
col3, col4 = st.columns(2)

with col3:
    suicidal_thoughts = st.selectbox("Have you ever had suicidal thoughts?", ["No", "Yes"])
    family_history = st.selectbox("Family History of Mental Illness", ["No", "Yes"])

with col4:
    dietary_habit = st.selectbox("Select Dietary Habit", ["Healthy", "Moderate", "Unhealthy"])

# Create input dictionary
inputs = {
    "Gender": 1 if gender == "Female" else 0,
    "Working Professional or Student": 1 if role == "Working Professional" else 0,
    "City": city,
    "Profession": profession,
    "Degree": degree,
    "Sleep Duration": sleep_duration,
    "Have you ever had suicidal thoughts ?": 1 if suicidal_thoughts == "Yes" else 0,
    "Family History of Mental Illness": 1 if family_history == "Yes" else 0,
    "Dietary Habits_Healthy": 1 if dietary_habit == "Healthy" else 0,
    "Dietary Habits_Moderate": 1 if dietary_habit == "Moderate" else 0,
    "Dietary Habits_Unhealthy": 1 if dietary_habit == "Unhealthy" else 0,
    "Age": age,
    "Academic Pressure": academic_pressure,
    "Work Pressure": work_pressure,
    "Study Satisfaction": study_satisfaction,
    "Job Satisfaction": job_satisfaction,
    "Work/Study Hours": work_study_hours,
    "Financial Stress": financial_stress
}

# Display current selections
st.subheader("Current Selections")
st.write("Please review your selections before prediction:")
col5, col6 = st.columns(2)

with col5:
    st.write("Personal Details:")
    st.write(f"- Gender: {gender}")
    st.write(f"- Role: {role}")
    st.write(f"- City: {city}")
    st.write(f"- Profession: {profession}")
    st.write(f"- Degree: {degree}")
    st.write(f"- Age: {age}")
    st.write(f"- Sleep Duration: {sleep_duration}")

with col6:
    st.write("Health & Stress Indicators:")
    if role == "Student":
        st.write(f"- Academic Pressure: {academic_pressure}")
        st.write(f"- Study Satisfaction: {study_satisfaction}")
    else:
        st.write(f"- Work Pressure: {work_pressure}")
        st.write(f"- Job Satisfaction: {job_satisfaction}")
    st.write(f"- Work/Study Hours: {work_study_hours}")
    st.write(f"- Financial Stress: {financial_stress}")
    st.write(f"- Dietary Habits: {dietary_habit}")
    st.write(f"- Family History: {family_history}")
    st.write(f"- Suicidal Thoughts: {suicidal_thoughts}")

# Prediction button and results
if st.button("Predict"):
    input_data = pd.DataFrame([inputs])
    input_data = input_data[catboost_model.feature_names_]
    
    categorical_features = ['City', 'Profession', 'Degree', 'Sleep Duration']
    input_pool = Pool(data=input_data, cat_features=categorical_features)
    
    prediction = catboost_model.predict(input_pool)
    
    st.subheader("Prediction Result")
    if prediction[0] == 0:
        st.success("Result: No indication of depression detected")
    else:
        st.warning("Result: Potential risk of depression detected")
        st.info("Please consult with a mental health professional for proper evaluation and support.")