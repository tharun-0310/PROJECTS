import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.impute import SimpleImputer
import joblib
import warnings

warnings.simplefilter(action="ignore")

# Load the data
data = pd.read_csv(r'F:\BDA prj\diabetes_prediction_dataset.csv')
df = data

# Display the first few rows and basic information about the dataset
print(data.head())
print(data.info())
# Check for missing values
print(data.isnull().sum())

# Encode categorical variables
df['gender'] = df['gender'].map({'Female': 0, 'Male': 1})
df['smoking_history'] = df['smoking_history'].astype('category').cat.codes

# Descriptive statistics of the data set
print(df.describe([0.10,0.25,0.50,0.75,0.90,0.95,0.99]).T)
# Distribution of the diabetes variable
print(df["diabetes"].value_counts()*100/len(df))

# Correlation matrix
plt.figure(figsize=[20,15])
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="magma")
plt.title("Correlation Matrix", fontsize=20)
plt.show()

# Data Preprocessing
X = df.drop("diabetes", axis=1)
y = df["diabetes"]

# Standardization
scaler = StandardScaler()
X = scaler.fit_transform(X)
X = pd.DataFrame(X, columns=df.columns.drop("diabetes"))

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training and evaluation
models = [
    ('LR', LogisticRegression(random_state=42)),
    ('KNN', KNeighborsClassifier()),
    ('CART', DecisionTreeClassifier(random_state=42)),
    ('RF', RandomForestClassifier(random_state=42)),
    ('SVM', SVC(gamma='auto', random_state=42)),
    ('XGB', XGBClassifier(random_state=42)),
]

results = []
names = []

# Imputer for missing values
imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

for name, model in models:
    kfold = KFold(n_splits=10, random_state=42, shuffle=True)
    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring="accuracy")
    results.append(cv_results)
    names.append(name)
    print(f"{name}: {cv_results.mean():.4f} ({cv_results.std():.4f})")

# Boxplot algorithm comparison
plt.figure(figsize=(15,10))
plt.title('Algorithm Comparison')
plt.boxplot(results)
plt.xticks(range(1, len(names) + 1), names)
plt.ylabel('Accuracy')
plt.show()

# XGBoost model
xgb = XGBClassifier(random_state=42)
xgb.fit(X_train, y_train)
y_pred = xgb.predict(X_test)

# Save the trained XGBoost model
joblib.dump(xgb, 'xgb_model.pkl')
print("Model saved as 'xgb_model.pkl'")

print("\nXGBoost Performance:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Feature importance
feature_importance = xgb.feature_importances_
feature_importance_df = pd.DataFrame({'feature': X.columns, 'importance': feature_importance})
feature_importance_df = feature_importance_df.sort_values('importance', ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(x='importance', y='feature', data=feature_importance_df)
plt.title('Feature Importance (XGBoost)')
plt.tight_layout()
plt.show()

# ROC Curve
y_pred_proba = xgb.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
auc = roc_auc_score(y_test, y_pred_proba)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()

# Additional visualizations
# 1. Age Distribution
plt.figure(figsize=(12, 6))
sns.histplot(data=df, x='age', hue='diabetes', kde=True, multiple="stack")
plt.title('Age Distribution by Diabetes Status')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()

# 2. BMI Distribution
plt.figure(figsize=(12, 6))
sns.histplot(data=df, x='bmi', hue='diabetes', kde=True, multiple="stack")
plt.title('BMI Distribution by Diabetes Status')
plt.xlabel('BMI')
plt.ylabel('Count')
plt.show()

# 3. HbA1c Level vs Blood Glucose Level
plt.figure(figsize=(12, 8))
sns.scatterplot(data=df, x='HbA1c_level', y='blood_glucose_level', hue='diabetes')
plt.title('HbA1c Level vs Blood Glucose Level')
plt.xlabel('HbA1c Level')
plt.ylabel('Blood Glucose Level')
plt.show()

# 4. Gender and Diabetes
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='gender', hue='diabetes')
plt.title('Diabetes Prevalence by Gender')
plt.xlabel('Gender (0: Female, 1: Male)')
plt.ylabel('Count')
plt.show()

# 5. Smoking History and Diabetes
plt.figure(figsize=(12, 6))
sns.countplot(data=df, x='smoking_history', hue='diabetes')
plt.title('Diabetes Prevalence by Smoking History')
plt.xlabel('Smoking History')
plt.ylabel('Count')
plt.show()

# 6. Age vs BMI with Diabetes
plt.figure(figsize=(12, 8))
sns.scatterplot(data=df, x='age', y='bmi', hue='diabetes', size='blood_glucose_level', sizes=(20, 200))
plt.title('Age vs BMI with Diabetes and Blood Glucose Level')
plt.xlabel('Age')
plt.ylabel('BMI')
plt.show()

# 7. Correlation Heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()

# 8. Pairplot
sns.pairplot(df, hue='diabetes', vars=['age', 'bmi', 'HbA1c_level', 'blood_glucose_level'])
plt.suptitle('Pairplot of Key Variables', y=1.02)
plt.show()
