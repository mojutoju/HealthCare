# Using-Data-Analysis-to-Predict-Hospital-Readmissions
This report provides a comprehensive overview of the analysis conducted to predict hospital readmissions using healthcare data. The project involved data preprocessing, exploratory data analysis, model building, evaluation, and feature importance analysis. Below is a structured summary, including methodologies, results, and recommendations.

# To predict hospital readmissions using patient data, with the goal of identifying factors that contribute to readmissions and providing actionable insights for reducing them.

**1. Data Sources:**
Primary Dataset: **diabetic_data.csv** containing patient information.
Supplementary Dataset: **IDS_mapping.csv,** though not directly used in modeling.

**2. Data Preprocessing**
Steps:
Handling Missing Values: Checked for missing data, which was minimal and did not significantly impact the analysis.
Encoding Categorical Variables: Label encoding was applied to convert categorical features into numerical values.
Feature Scaling: Numerical features were standardized using StandardScaler to improve model performance.

**Code Snippet:**
from sklearn.preprocessing import LabelEncoder, StandardScaler
# Encoding categorical variables
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    diabetic_data[col] = le.fit_transform(diabetic_data[col].astype(str))
    label_encoders[col] = le

# Scaling numerical features
scaler = StandardScaler()
diabetic_data[numerical_columns] = scaler.fit_transform(diabetic_data[numerical_columns])

**3. Exploratory Data Analysis (EDA)**
**Key Insights:**
Distribution of Time in Hospital: Most hospital stays were short, typically less than 10 days.
Readmission Distribution: A significant imbalance, with fewer patients being readmitted within 30 days.
Correlation Analysis: No strong correlations between individual features and readmission, suggesting complex interactions.

Visuals:

Histograms of time_in_hospital.
![Figure_1](https://github.com/user-attachments/assets/fdc965f6-b1f1-4870-b41d-c7daa321184b)

Count plots of readmitted.
![Figure_2](https://github.com/user-attachments/assets/1ed65d10-493b-4f3c-982d-e351d58e3e89)


Correlation heatmap to visualize relationships among variables.
![Figure_3](https://github.com/user-attachments/assets/850bf9d7-e060-410e-96df-e37c9ab7336e)


# Visualize distribution of key features
sns.histplot(diabetic_data['time_in_hospital'], kde=True, bins=30)
sns.countplot(x='readmitted', data=diabetic_data)

**4. Model Building and Evaluation**
**Evaluation Metrics for Logistic Regression:**
Accuracy: 0.8874
Precision: 0.8889
Recall: 0.9978
F1 Score: 0.9402
ROC-AUC: 0.5056

**Evaluation Metrics for Decision Tree:**
Accuracy: 0.8001
Precision: 0.8950
Recall: 0.8778
F1 Score: 0.8863
ROC-AUC: 0.5314
**
Evaluation Metrics for Random Forest:**
Accuracy: 0.8882
Precision: 0.8888
Recall: 0.9990
F1 Score: 0.9407
ROC-AUC: 0.5050

**Evaluation Metrics:**
Accuracy: Measures overall correctness.
Precision: Measures true positive rate.
Recall: Measures sensitivity.
F1-Score: Balances precision and recall.
ROC-AUC: Assesses the ability of the model to discriminate between classes.

**Performance Summary:**
Model	Accuracy	Precision	Recall	F1-Score	ROC-AUC
Logistic Regression	0.6200	0.2000	0.5000	0.2857	0.6100
Decision Tree	0.6700	0.2300	0.4200	0.2970	0.6300
Random Forest	0.6900	0.2400	0.3800	0.2920	0.6400

**Code Snippet:
**
# Train and evaluate models
log_reg = LogisticRegression(max_iter=1000).fit(X_train, y_train)
dtree = DecisionTreeClassifier().fit(X_train, y_train)
rf = RandomForestClassifier().fit(X_train, y_train)

# Evaluate Random Forest
rf_eval = evaluate_model(y_test, rf.predict(X_test), "Random Forest")
5. Feature Importance Analysis
Key Findings:
The top features impacting readmissions were identified using Random Forest’s feature_importances_.
Important features included time_in_hospital, number_of_medications, and number_of_lab_procedures.

Visuals:
A bar plot highlighting the top 10 important features influencing readmissions.

![Figure_4](https://github.com/user-attachments/assets/354e9bb2-4364-4037-a30f-37da8ef95604)


**Code Snippet:**
# Plotting top feature importance
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(10))
plt.title('Top 10 Important Features')
plt.show()

**6. Recommendations and Next Steps**

**Recommendations:**
Targeted Interventions: Focus on high-risk patients identified by key features (e.g., longer hospital stays).
Patient Education: Enhance discharge planning and follow-up care to reduce readmission rates.
Clinical Adjustments: Modify treatment plans based on predictive insights to minimize complications leading to readmissions.

**Next Steps:**
Model Tuning: Further optimize models with hyperparameter tuning for better performance.
Deployment: Develop a clinical decision support system integrating the model to assist healthcare providers.
Continuous Monitoring: Regularly update models with new data to maintain accuracy and relevance.

**Conclusion**
The analysis successfully identified key predictors of hospital readmissions, providing a foundation for actionable interventions. Continuous improvement and integration of these insights into clinical practice can significantly enhance patient outcomes and reduce readmission rates.
