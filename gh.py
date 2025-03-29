import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import datetime

# Load existing data or create a new DataFrame
dataframe = r"pythonProject\fitoproto\height_weight_data.csv"
if 'bmi_data' not in st.session_state:
    st.session_state.bmi_data = pd.DataFrame(columns=['Date', 'Weight (kg)', 'Height (cm)'])

# Title
st.title("Weight Tracker")

# User input
weight = st.number_input("Enter your weight (kg):", min_value=30.0, max_value=200.0, step=0.1)
height = st.number_input("Enter your height (cm):", min_value=100.0, max_value=250.0, step=0.1)

today = datetime.date.today()

if st.button("Log Data"):
    new_entry = pd.DataFrame({
        'Date': [today],
        'Weight (kg)': [weight],
        'Height (cm)': [height]
    })
    
    st.session_state.bmi_data = pd.concat([st.session_state.bmi_data, new_entry], ignore_index=True)
    st.success(f"Logged Weight: {weight} kg and Height: {height} cm")

# Ensure 'Date' is in datetime format
st.session_state.bmi_data['Date'] = pd.to_datetime(st.session_state.bmi_data['Date'])

# Sort values by date
st.session_state.bmi_data = st.session_state.bmi_data.sort_values(by='Date')

# Display Data
if not st.session_state.bmi_data.empty:
    st.subheader("Logged Data")
    st.dataframe(st.session_state.bmi_data)
    
    # Debug: Check column names
    st.write("Columns in Data:", st.session_state.bmi_data.columns.tolist())

    # Plot Weight Trend
    st.subheader("Weight Trend")
    fig, ax = plt.subplots()
    ax.set_xlabel("Date")
    ax.set_ylabel("Weight (kg)", color='tab:blue')
    ax.plot(st.session_state.bmi_data['Date'], st.session_state.bmi_data['Weight (kg)'], marker='o', linestyle='-', color='tab:blue', label='Weight')
    ax.tick_params(axis='y', labelcolor='tab:blue')
    ax.grid(True)  # Add grid for better readability

    fig.autofmt_xdate()  # Format date labels
    st.pyplot(fig)

    # Plot Weight vs Height
    st.subheader("Weight vs Height")
    fig, ax = plt.subplots()
    ax.set_xlabel("Height (cm)")
    ax.set_ylabel("Weight (kg)", color='tab:red')

    if not st.session_state.bmi_data[['Height (cm)', 'Weight (kg)']].isnull().values.any():
        ax.scatter(st.session_state.bmi_data['Height (cm)'], st.session_state.bmi_data['Weight (kg)'], color='tab:red', label='Weight vs Height')
        ax.legend()
    else:
        st.error("Error: Missing data for Height and Weight.")

    st.pyplot(fig)
