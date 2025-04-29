# import streamlit as st
# import pandas as pd
# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import accuracy_score, classification_report, roc_curve, roc_auc_score
# from xgboost import XGBClassifier
# import shap
# import joblib

# # Load the dataset
# st.title("Diabetes Prediction App")

# # Load the data
# data_path = r'F:\BDA prj\diabetes_prediction_dataset.csv'
# data = pd.read_csv(data_path)
# df = data.copy()

# # Preprocessing
# df['gender'] = df['gender'].map({'Female': 0, 'Male': 1})
# df['smoking_history'] = df['smoking_history'].astype('category').cat.codes

# # Separate features and target
# X = df.drop("diabetes", axis=1)
# y = df["diabetes"]

# # Standardize the data
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
# X_scaled = pd.DataFrame(X_scaled, columns=X.columns)  # Ensure correct feature names are maintained

# # Split the data
# X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# # Sidebar - User Inputs for Prediction
# st.sidebar.header('User Input Parameters')
# def user_input_features():
#     age = st.sidebar.slider('Age', int(df['age'].min()), int(df['age'].max()), int(df['age'].mean()))
#     bmi = st.sidebar.slider('BMI', float(df['bmi'].min()), float(df['bmi'].max()), float(df['bmi'].mean()))
#     gender = st.sidebar.selectbox('Gender', (0, 1))
#     HbA1c_level = st.sidebar.slider('HbA1c Level', float(df['HbA1c_level'].min()), float(df['HbA1c_level'].max()), float(df['HbA1c_level'].mean()))
#     blood_glucose_level = st.sidebar.slider('Blood Glucose Level', float(df['blood_glucose_level'].min()), float(df['blood_glucose_level'].max()), float(df['blood_glucose_level'].mean()))
#     smoking_history = st.sidebar.selectbox('Smoking History', list(df['smoking_history'].unique()))
#     heart_disease = st.sidebar.selectbox('Heart Disease', (0, 1))
#     hypertension = st.sidebar.selectbox('Hypertension', (0, 1))

#     data = {'age': age,
#             'bmi': bmi,
#             'gender': gender,
#             'HbA1c_level': HbA1c_level,
#             'blood_glucose_level': blood_glucose_level,
#             'smoking_history': smoking_history,
#             'heart_disease': heart_disease,
#             'hypertension': hypertension}
    
#     features = pd.DataFrame(data, index=[0])
#     return features

# user_input_df = user_input_features()

# # Display user inputs
# st.subheader('Specified Input parameters')
# st.write(user_input_df)

# # Model Training
# model = RandomForestClassifier(random_state=42)
# model.fit(X_train, y_train)

# # Prediction
# # Align user input with training features and scale
# user_input_scaled = scaler.transform(user_input_df[X.columns])  
# prediction = model.predict(user_input_scaled)
# prediction_proba = model.predict_proba(user_input_scaled)

# # Display Prediction Result
# st.subheader('Prediction')
# st.write('Diabetes' if prediction[0] else 'No Diabetes')

# st.subheader('Prediction Probability')
# st.write(prediction_proba)

# # Model Evaluation
# y_pred = model.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# st.subheader('Model Accuracy')
# st.write(f'Accuracy: {accuracy:.2f}')

# st.subheader('Classification Report')
# st.text(classification_report(y_test, y_pred))

# # Feature Importance using SHAP
# st.header('Feature Importance')
# explainer = shap.TreeExplainer(model)
# shap_values = explainer.shap_values(X_train)

# plt.title('Feature Importance based on SHAP values')
# shap.summary_plot(shap_values, X_train, feature_names=X.columns)
# st.pyplot(bbox_inches='tight')

# # ROC Curve
# st.header("ROC Curve")
# y_pred_proba = model.predict_proba(X_test)[:, 1]
# fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
# auc_score = roc_auc_score(y_test, y_pred_proba)

# plt.figure(figsize=(10, 6))
# plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.2f})')
# plt.plot([0, 1], [0, 1], linestyle='--')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic (ROC) Curve')
# plt.legend(loc="lower right")
# st.pyplot()

# # Save the model
# joblib.dump(model, 'diabetes_rf_model.pkl')
# st.write("Model saved as 'diabetes_rf_model.pkl'")

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_curve, roc_auc_score
from xgboost import XGBClassifier
import shap
import joblib

# Load the dataset
st.title("Diabetes Prediction App")

# Load the data
data_path = r'F:\BDA prj\diabetes_prediction_dataset.csv'
data = pd.read_csv(data_path)
df = data.copy()

# Preprocessing
df['gender'] = df['gender'].map({'Female': 0, 'Male': 1})
df['smoking_history'] = df['smoking_history'].astype('category').cat.codes

# Separate features and target
X = df.drop("diabetes", axis=1)
y = df["diabetes"]

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Sidebar - User Inputs for Prediction
st.sidebar.header('User Input Parameters')
def user_input_features():
    age = st.sidebar.slider('Age', int(df['age'].min()), int(df['age'].max()), int(df['age'].mean()))
    bmi = st.sidebar.slider('BMI', float(df['bmi'].min()), float(df['bmi'].max()), float(df['bmi'].mean()))
    gender = st.sidebar.selectbox('Gender', (0, 1))
    HbA1c_level = st.sidebar.slider('HbA1c Level', float(df['HbA1c_level'].min()), float(df['HbA1c_level'].max()), float(df['HbA1c_level'].mean()))
    blood_glucose_level = st.sidebar.slider('Blood Glucose Level', float(df['blood_glucose_level'].min()), float(df['blood_glucose_level'].max()), float(df['blood_glucose_level'].mean()))
    smoking_history = st.sidebar.selectbox('Smoking History', list(df['smoking_history'].unique()))
    heart_disease = st.sidebar.selectbox('Heart Disease', (0, 1))
    hypertension = st.sidebar.selectbox('Hypertension', (0, 1))

    data = {'age': age,
            'bmi': bmi,
            'gender': gender,
            'HbA1c_level': HbA1c_level,
            'blood_glucose_level': blood_glucose_level,
            'smoking_history': smoking_history,
            'heart_disease': heart_disease,
            'hypertension': hypertension}
    
    # Ensure the order of columns matches the training data
    features = pd.DataFrame(data, index=[0])
    features = features[X.columns]  # Align column order with X used in training
    return features

user_input_df = user_input_features()

# Display user inputs
st.subheader('Specified Input parameters')
st.write(user_input_df)

# Model Training
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Prediction
user_input_scaled = scaler.transform(user_input_df)  # Transform user input data using the same scaler
prediction = model.predict(user_input_scaled)
prediction_proba = model.predict_proba(user_input_scaled)

# Display Prediction Result
st.subheader('Prediction')
st.write('Diabetes' if prediction[0] else 'No Diabetes')

st.subheader('Prediction Probability')
st.write(prediction_proba)

# Model Evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
st.subheader('Model Accuracy')
st.write(f'Accuracy: {accuracy:.2f}')

st.subheader('Classification Report')
st.text(classification_report(y_test, y_pred))

# Feature Importance using SHAP
st.header('Feature Importance')
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_train)

plt.title('Feature Importance based on SHAP values')
shap.summary_plot(shap_values, X_train, feature_names=X.columns)
st.pyplot(bbox_inches='tight')

# ROC Curve
st.header("ROC Curve")
y_pred_proba = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
auc_score = roc_auc_score(y_test, y_pred_proba)

plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
st.pyplot()

# Save the model
joblib.dump(model, 'diabetes_rf_model.pkl')
st.write("Model saved as 'diabetes_rf_model.pkl'")
