import streamlit as st
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load dataset
df = sns.load_dataset('titanic')
df['Attrition'] = df['survived']
df = df.drop(['survived'], axis=1)
df = df.dropna()

# Convert categorical
df = pd.get_dummies(df)

# Split
X = df.drop('Attrition', axis=1)
y = df['Attrition']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# UI
st.title("Employee Attrition Prediction System")

st.sidebar.header("Enter Details")
age = st.sidebar.slider("Age", 18, 60, 25)
fare = st.sidebar.slider("Salary", 10, 500, 100)

input_data = X_test.iloc[0:1].copy()
input_data[:] = 0
input_data['age'] = age
input_data['fare'] = fare

prediction = model.predict(input_data)
prob = model.predict_proba(input_data)[0][1]

if prob < 0.3:
    risk = "Low Risk"
elif prob < 0.6:
    risk = "Medium Risk"
else:
    risk = "High Risk"

st.subheader("Result")
st.write("Prediction:", "Will Leave" if prediction[0]==1 else "Will Stay")
st.write("Probability:", prob)
st.write("Risk Level:", risk)

st.bar_chart(df['Attrition'].value_counts())
