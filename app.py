import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Simple dataset
data = {
    'Age': [25, 30, 45, 35, 22, 40],
    'Salary': [30000, 40000, 80000, 50000, 20000, 90000],
    'Attrition': [0, 0, 1, 0, 1, 1]
}

df = pd.DataFrame(data)

X = df[['Age', 'Salary']]
y = df['Attrition']

model = RandomForestClassifier()
model.fit(X, y)

st.title("Employee Attrition Prediction System")

age = st.slider("Age", 18, 60, 25)
salary = st.slider("Salary", 10000, 100000, 30000)

input_data = pd.DataFrame([[age, salary]], columns=['Age', 'Salary'])

prediction = model.predict(input_data)[0]
prob = model.predict_proba(input_data)[0][1]

if prob < 0.3:
    risk = "Low Risk"
elif prob < 0.6:
    risk = "Medium Risk"
else:
    risk = "High Risk"

st.write("Prediction:", "Will Leave" if prediction == 1 else "Will Stay")
st.write("Probability:", prob)
st.write("Risk Level:", risk)

