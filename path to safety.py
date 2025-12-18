# =========================================
# Path To Safety - Road Traffic Accident ML
# =========================================

# 1. Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# 2. Load Dataset
# Ensure road_accident_data.csv is in the same directory
data = pd.read_csv("road_accident_data.csv")

print("First 5 rows of dataset:")
print(data.head())


# 3. Data Preprocessing

# Handle missing values
data.fillna(method='ffill', inplace=True)

# Encode categorical features
le = LabelEncoder()
categorical_cols = [
    'Weather',
    'Road_Type',
    'Light_Condition',
    'Vehicle_Type',
    'Time_of_Day'
]

for col in categorical_cols:
    data[col] = le.fit_transform(data[col])

print("\nDataset info after preprocessing:")
print(data.info())


# 4. Exploratory Data Analysis (EDA)

# Accident severity distribution
plt.figure(figsize=(6,4))
sns.countplot(x='Accident_Severity', data=data)
plt.title("Accident Severity Distribution")
plt.show()

# Accidents by weather condition
plt.figure(figsize=(6,4))
sns.countplot(x='Weather', hue='Accident_Severity', data=data)
plt.title("Accidents by Weather Condition")
plt.show()

# Accidents by road type
plt.figure(figsize=(6,4))
sns.countplot(x='Road_Type', hue='Accident_Severity', data=data)
plt.title("Accidents by Road Type")
plt.show()

# Correlation heatmap
plt.figure(figsize=(10,8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()


# 5. Feature Selection
X = data.drop('Accident_Severity', axis=1)
y = data['Accident_Severity']


# 6. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# =========================
# 7. Model Training
# =========================

# Decision Tree Classifier
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

dt_pred = dt.predict(X_test)

print("\n--- Decision Tree Model ---")
print("Accuracy:", accuracy_score(y_test, dt_pred))
print(classification_report(y_test, dt_pred))


# Random Forest Classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)

print("\n--- Random Forest Model ---")
print("Accuracy:", accuracy_score(y_test, rf_pred))
print(classification_report(y_test, rf_pred))


# 8. Confusion Matrix (Random Forest)
cm = confusion_matrix(y_test, rf_pred)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# 9. Feature Importance (Random Forest)
importances = rf.feature_importances_
feature_names = X.columns

plt.figure(figsize=(8,5))
sns.barplot(x=importances, y=feature_names)
plt.title("Feature Importance in Accident Prediction")
plt.show()


# 10. Prediction on New Accident Scenario
# Example input:
# [Speed, Weather, Road_Type, Light_Condition, Vehicle_Type, Time_of_Day]

new_data = np.array([[80, 1, 2, 0, 1, 3]])
prediction = rf.predict(new_data)

print("\nNew Scenario Prediction:")
if prediction[0] == 2:
    print("High Risk / Fatal Accident Scenario")
elif prediction[0] == 1:
    print("Moderate Risk Accident Scenario")
else:
    print("Low Risk / Minor Accident Scenario")  