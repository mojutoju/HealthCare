# Step 1: Load necessary libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report

# Load the datasets
diabetic_data = pd.read_csv('diabetic_data.csv')
ids_mapping = pd.read_csv('IDS_mapping.csv')

# Display the first few rows of each dataset
print("Diabetic Data Head:")
print(diabetic_data.head())

print("\nIDS Mapping Data Head:")
print(ids_mapping.head())

# Display basic information about the datasets
print("\nDiabetic Data Info:")
print(diabetic_data.info())

print("\nIDS Mapping Data Info:")
print(ids_mapping.info())


# Step 2: Data Preprocessing

# Import necessary libraries
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

# 1. Handle Missing Values
# Checking missing values in diabetic data
missing_values = diabetic_data.isnull().sum()
print("Missing Values in Diabetic Data:\n", missing_values[missing_values > 0])

# Filling missing values or removing rows with missing values based on the importance
# Example: Dropping rows with missing 'weight' column if needed, or imputing with median/mode
# diabetic_data['weight'].fillna(diabetic_data['weight'].median(), inplace=True) # Example for imputation

# 2. Merge Datasets if Required
# Checking if there's a key column to merge on, like 'encounter_id' or 'patient_nbr'
# Example: diabetic_data = pd.merge(diabetic_data, ids_mapping, how='left', on='common_key_column')

# 3. Encode Categorical Variables
# List of columns that are categorical
categorical_columns = diabetic_data.select_dtypes(include=['object']).columns

# Apply Label Encoding or One-Hot Encoding to categorical variables
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    diabetic_data[col] = le.fit_transform(diabetic_data[col].astype(str))
    label_encoders[col] = le

# 4. Feature Scaling
# Scaling numerical features to have a mean of 0 and standard deviation of 1
scaler = StandardScaler()
numerical_columns = diabetic_data.select_dtypes(include=['int64', 'float64']).columns
diabetic_data[numerical_columns] = scaler.fit_transform(diabetic_data[numerical_columns])

# Display the preprocessed data
print("\nPreprocessed Data Sample:")
print(diabetic_data.head())

# 1. Handle Missing Values
# Checking missing values in diabetic data
missing_values = diabetic_data.isnull().sum()

# Filling missing values or handling them accordingly (for simplicity, no specific handling applied yet)
# Example of missing value handling could be added based on further requirements.

# 2. Merge Datasets if Required (not merging here as no specific key was found to merge)
# This step is skipped as it seems IDS_mapping is supplementary data.

# 3. Encode Categorical Variables
# List of columns that are categorical
categorical_columns = diabetic_data.select_dtypes(include=['object']).columns

# Apply Label Encoding to categorical variables
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    diabetic_data[col] = le.fit_transform(diabetic_data[col].astype(str))
    label_encoders[col] = le

# 4. Feature Scaling
# Scaling numerical features to have a mean of 0 and standard deviation of 1
scaler = StandardScaler()
numerical_columns = diabetic_data.select_dtypes(include=['int64', 'float64']).columns
diabetic_data[numerical_columns] = scaler.fit_transform(diabetic_data[numerical_columns])

# Display the preprocessed data
preprocessed_sample = diabetic_data.head()

print(missing_values, preprocessed_sample)


# 1. Visualize Distribution of Time in Hospital
plt.figure(figsize=(14, 6))
sns.histplot(diabetic_data['time_in_hospital'], kde=True, bins=30, color='blue')
plt.title('Distribution of Time in Hospital')
plt.xlabel('Time in Hospital (Days)')
plt.ylabel('Frequency')
plt.show()

# 2. Visualize Distribution of Readmission
plt.figure(figsize=(10, 5))
sns.countplot(x='readmitted', data=diabetic_data)
plt.title('Readmission Distribution')
plt.xlabel('Readmitted (0 = No, 1 = Yes)')
plt.ylabel('Count')
plt.show()

# 3. Correlation Analysis
plt.figure(figsize=(14, 10))
correlation = diabetic_data.corr()
sns.heatmap(correlation, annot=False, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Display top correlations with the target variable 'readmitted'
top_correlations = correlation['readmitted'].sort_values(ascending=False).head(10)
print('Top Correlations')
print(top_correlations)


# Step 4: Model Building and Evaluation



# 1. Split the data into training and testing sets
X = diabetic_data.drop(columns=['readmitted'])  # Features
y = (diabetic_data['readmitted'] > 0).astype(int)        # Target

# Splitting the data into 70% training and 30% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 2. Train Logistic Regression Model
log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train, y_train)
log_reg_pred = log_reg.predict(X_test)

# 3. Train Decision Tree Model
dtree = DecisionTreeClassifier(random_state=42)
dtree.fit(X_train, y_train)
dtree_pred = dtree.predict(X_test)

# 4. Train Random Forest Model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

# 5. Evaluate Models using various metrics
def evaluate_model(y_test, y_pred, model_name):
    print(f"Evaluation Metrics for {model_name}:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred, average='binary'):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred, average='binary'):.4f}")
    print(f"F1 Score: {f1_score(y_test, y_pred, average='binary'):.4f}")
    print(f"ROC-AUC: {roc_auc_score(y_test, y_pred):.4f}\n")

# Evaluate each model
evaluate_model(y_test, log_reg_pred, "Logistic Regression")
evaluate_model(y_test, dtree_pred, "Decision Tree")
evaluate_model(y_test, rf_pred, "Random Forest")


# Feature Importance for Random Forest
importances = rf.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Plotting Feature Importance
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(10))
plt.title('Top 10 Important Features in Random Forest')
plt.show()
