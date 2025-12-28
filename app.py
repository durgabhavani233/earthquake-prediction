import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(
    page_title="Earthquake Prediction App",
    layout="wide"
)

# --------------------------------------------------
# Title
# --------------------------------------------------
st.title("🌍 Global Earthquake Magnitude Prediction")
st.write("Machine Learning based Earthquake Magnitude Prediction App")

# --------------------------------------------------
# Load Dataset
# --------------------------------------------------
st.header("Step 1: Load Dataset")

@st.cache_data
def load_data():
    return pd.read_csv("earthquakes.csv")

data = load_data()
st.success("Dataset loaded successfully!")

if st.checkbox("Show raw dataset"):
    st.dataframe(data.head())

# --------------------------------------------------
# Data Preprocessing
# --------------------------------------------------
st.header("Step 2: Data Preprocessing")

data.fillna(0, inplace=True)

if 'time' in data.columns:
    data['time'] = pd.to_datetime(data['time'])
    data['year'] = data['time'].dt.year

st.write("✔ Missing values handled")
st.write("✔ Date-time converted")

# --------------------------------------------------
# Feature Engineering
# --------------------------------------------------
st.header("Step 3: Feature Engineering")

data['depth_category'] = pd.cut(
    data['depth'],
    bins=[0, 70, 300, 700],
    labels=['Shallow', 'Intermediate', 'Deep']
)

data['magnitude_severity'] = pd.cut(
    data['magnitude'],
    bins=[0, 4, 6, 10],
    labels=['Low', 'Medium', 'High']
)

data = pd.get_dummies(
    data,
    columns=['depth_category', 'magnitude_severity'],
    drop_first=True
)

st.write("✔ New features created")

# --------------------------------------------------
# EDA
# --------------------------------------------------
st.header("Step 4: Exploratory Data Analysis")

fig, ax = plt.subplots()
ax.hist(data['magnitude'], bins=20)
ax.set_title("Earthquake Magnitude Distribution")
ax.set_xlabel("Magnitude")
ax.set_ylabel("Frequency")
st.pyplot(fig)

# --------------------------------------------------
# Feature Selection
# --------------------------------------------------
st.header("Step 5: Feature Selection")

features = ['latitude', 'longitude', 'depth']
target = 'magnitude'

X = data[features]
y = data[target]

st.write("Selected Features:", features)

# --------------------------------------------------
# Train-Test Split & Scaling
# --------------------------------------------------
st.header("Step 6: Train-Test Split & Scaling")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

st.write("✔ Data split and scaled")

# --------------------------------------------------
# Model Training & Evaluation
# --------------------------------------------------
st.header("Step 7: Model Training & Evaluation")

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree Regressor": DecisionTreeRegressor(),
    "Random Forest Regressor": RandomForestRegressor(n_estimators=100),
    "Support Vector Regressor": SVR()
}

results = {}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    results[name] = round(r2_score(y_test, y_pred) * 100, 2)

results_df = pd.DataFrame(
    list(results.items()),
    columns=["Model", "R2 Score (%)"]
)

st.dataframe(results_df)

# --------------------------------------------------
# Select Best Model
# --------------------------------------------------
best_model_name = results_df.loc[
    results_df["R2 Score (%)"].idxmax(), "Model"
]

best_model = models[best_model_name]
st.success(f"Best Model Selected: {best_model_name}")

# --------------------------------------------------
# Prediction Section
# --------------------------------------------------
st.header("Step 8: Predict Earthquake Magnitude")

depth = st.number_input("Depth (km)", min_value=0.0, value=10.0)
latitude = st.number_input("Latitude", min_value=-90.0, max_value=90.0, value=15.0)
longitude = st.number_input("Longitude", min_value=-180.0, max_value=180.0, value=75.0)

if st.button("Predict"):
    input_data = np.array([[latitude, longitude, depth]])
    input_scaled = scaler.transform(input_data)
    prediction = best_model.predict(input_scaled)
    st.success(f"Predicted Earthquake Magnitude: {prediction[0]:.2f}")

# --------------------------------------------------
# Save Model
# --------------------------------------------------
st.header("Step 9: Save Model")

if st.button("Save Model & Scaler"):
    pickle.dump(best_model, open("earthquake_model.pkl", "wb"))
    pickle.dump(scaler, open("scaler.pkl", "wb"))
    st.success("Model and scaler saved successfully!")

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown("---")
st.markdown("**ML Mini Project – Global Earthquake Prediction**")
