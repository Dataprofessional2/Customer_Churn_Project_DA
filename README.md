# Customer Churn Prediction

This repository contains projects and case studies based on real-world business problems. It includes data cleaning, analysis, visualization, and insights using different tools. The goal is to apply real-world data analytics techniques to solve business problems and derive actionable insights.

## How Businesses Predict Customer Churn to Improve Customer Retention
## Table of Contents

- Introduction
- Overview of Customer Churn Dataset
- Business Objective
- Data Cleaning & Preparation
- Data Loading and Inspection
- Handling Missing Values
- Data Cleaning and Formatting
- Exploratory Data Analysis (EDA)
- Churn Distribution and Class Imbalance
- Customer Demographics and Churn Trends
- Contract Type and Payment Method Analysis
- Tenure and Monthly Charges Patterns
- Feature Engineering
- Modeling Approach
- Baseline Logistic Regression
- Advanced Model: Random Forest
- Model Evaluation
- Performance Metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- Insights and Recommendations
- Customer Retention Strategies
- Conclusion

## Project Overview: Customer Churn Prediction

### Introduction

In this project, we explore how businesses can identify customers who are likely to leave their service and take proactive steps to improve customer retention. Using a customer churn dataset, we analyze patterns in customer behavior, service usage, and billing information. The goal is to build a predictive model that helps organizations detect high-risk customers by leveraging available customer attributes and usage trends.


### Overview of Customer Churn Dataset

The dataset includes comprehensive information on customers, such as demographic details, services subscribed to, billing information, and account history. It provides insights into customer behavior and service usage patterns that influence customer retention and churn.

Key features in the dataset include:

- customerID: unique identifier for each customer
- gender: customer gender
- SeniorCitizen: indicates whether the customer is a senior citizen
- tenure: number of months the customer has stayed with the company
- PhoneService: whether the customer has phone service
- InternetService: type of internet service subscribed
- Contract: type of customer contract (monthly, yearly, etc.)
- MonthlyCharges: amount charged to the customer each month
- TotalCharges: total amount charged to the customer
- Churn: indicates whether the customer left the service (Yes/No) and more

This dataset allows us to analyze customer behavior, identify patterns related to churn, and build predictive models to support customer retention strategies.




## Business Objective

The primary objective of this project is to predict whether a customer is likely to churn based on features such as demographic information, service usage, contract type, and billing behavior. The target variable in this analysis is customer churn, where Churn = 1 indicates that the customer has left the service and Churn = 0 indicates that the customer has remained. By identifying patterns and relationships within the data, the project aims to help businesses detect high-risk customers early and implement effective retention strategies to reduce customer loss and improve overall customer satisfaction.


## Data Cleaning & Preparation

#### Data Loading and Inspection

- The first step involves loading the dataset using Python libraries such as pandas and inspecting its structure. We check the shape of the dataset, column types, and summary statistics to understand the data and identify areas that require cleaning.

#### Handling Missing Values

- Missing data is common in real-world datasets. We explore missing values across features such as TotalCharges, tenure, or service-related variables and decide on appropriate techniques such as imputation or row removal based on their impact on the analysis.

#### Data Cleaning and Formatting

- We ensure consistency in the dataset by converting data types, handling categorical variables, and removing duplicate records if necessary. This step prepares the data for efficient analysis and modeling.


```python
# Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```
![An Image](https://github.com/Dataprofessional2/Customer_Churn_Project_DA/blob/main/Importing_Libraries.png)  
```python
#Importing Churn DataSet
df=pd.read_csv("D://OneDrive//Documents//Telco-Customer-Churn.csv")
df
```
![An Image](https://github.com/Dataprofessional2/Customer_Churn_Project_DA/blob/main/Importing_Data.png) 

```python
#checking data
df.head()
```
![An Image](https://github.com/Dataprofessional2/Customer_Churn_Project_DA/blob/main/top_records.png)

```python
#Basic column view
df.shape
df.info()
```
![An Image](https://github.com/Dataprofessional2/Customer_Churn_Project_DA/blob/main/df_inspection.png)


```python
# Selecting relevant columns into a new dataframe instead of deleting columns
# to preserve the original dataset. This ensures data integrity and allows us
# to revisit the raw data if needed during later stages of analysis.

selected_columns = [
    'gender',
    'SeniorCitizen',
    'Partner',
    'Dependents',
    'tenure',
    'PhoneService',
    'MultipleLines',
    'InternetService',
    'OnlineSecurity',
    'OnlineBackup',
    'DeviceProtection',
    'TechSupport',
    'StreamingTV',
    'StreamingMovies',
    'Contract',
    'PaperlessBilling',
    'PaymentMethod',
    'MonthlyCharges',
    'TotalCharges',
    'Churn'
]
df1 = df[selected_columns]
df1
```
![An Image](https://github.com/Dataprofessional2/Customer_Churn_Project_DA/blob/main/selecting_columns.png)
![An Image](https://github.com/Dataprofessional2/Customer_Churn_Project_DA/blob/main/img.png)
```python
#Data Cleaning
#Checking for null values
df.isnull().sum()
```
![An Image](https://github.com/Dataprofessional2/Customer_Churn_Project_DA/blob/main/Data_Cleanig.png)

```python
#Check Unique or Inconsistent Values (Categorical Columns)
for col in df1.select_dtypes(include='object').columns:
    print(col)
    print(df1[col].unique())
    print("------")
```
![An Image](https://github.com/Dataprofessional2/Customer_Churn_Project_DA/blob/main/dc2.png)

```python
df1.describe()
```
![An Image](https://github.com/Dataprofessional2/Customer_Churn_Project_DA/blob/main/df3.png)
```python
#Data Cleaning
# Replace blank spaces with NaN
df1['TotalCharges'] = df1['TotalCharges'].replace([' ', ''], pd.NA)

# Convert TotalCharges to numeric
df1['TotalCharges'] = pd.to_numeric(df1['TotalCharges'], errors='coerce')

# Check missing values again
df1.isnull().sum()

# Remove rows with missing values

df1 = df1.dropna()
# Check null values
df1.isnull().sum()

# Check dataset shape
df1.shape

# Check data types
df1.dtypes

# Summary statistics
df1.describe()
```
![An Image](https://github.com/Dataprofessional2/Customer_Churn_Project_DA/blob/main/df4.png)
![An Image](https://github.com/Dataprofessional2/Customer_Churn_Project_DA/blob/main/dc5.png)



## Exploratory Data Analysis
```python
df1['Churn'].value_counts()
```
![An Image](https://github.com/Dataprofessional2/Customer_Churn_Project_DA/blob/main/churn_count.png)


```python
# Basic Churn Plot
sns.countplot(x='Churn', data=df1)

plt.title('Customer Churn Distribution')
plt.xlabel('Churn')
plt.ylabel('Number of Customers')

plt.show()
```
![An Image](https://github.com/Dataprofessional2/Customer_Churn_Project_DA/blob/main/churn_plot.png)

```python
# Churn by gender
df1.groupby(['gender', 'Churn']).size()
```
![An Image](https://github.com/Dataprofessional2/Customer_Churn_Project_DA/blob/main/gender_churn.png)

```python
#gender churn by plot
sns.countplot(x='gender', hue='Churn', data=df1)

plt.title('Customer Churn Count by Gender')
plt.xlabel('Gender')
plt.ylabel('Number of Customers')
plt.show()
```
![An Image](https://github.com/Dataprofessional2/Customer_Churn_Project_DA/blob/main/gender_plot.png)

```python
# Generate summary statistics of MonthlyCharges grouped by churn status
df1.groupby('Churn')['MonthlyCharges'].describe()df1.groupby('Churn')['MonthlyCharges'].describe()
```
![An Image](https://github.com/Dataprofessional2/Customer_Churn_Project_DA/blob/main/desc_churn.png)

```python
#monthly charges for churn 
df1.groupby('Churn')['MonthlyCharges'].mean()
#median price for churn and not churn
df1.groupby('Churn')['MonthlyCharges'].median()
```
![An Image](https://github.com/Dataprofessional2/Customer_Churn_Project_DA/blob/main/monthly_price_churn.png)
```python
#monthly price boxplot
sns.boxplot(x='Churn', y='MonthlyCharges', data=df1)

plt.title('Monthly Charges by Churn Status')
plt.xlabel('Churn')
plt.ylabel('Monthly Charges')

plt.show()
```
![An Image](https://github.com/Dataprofessional2/Customer_Churn_Project_DA/blob/main/boxplot.png)

```python
# Churn count by Senior Citizen
df1.groupby('SeniorCitizen')['Churn'].value_counts()
```
![An Image](https://github.com/Dataprofessional2/Customer_Churn_Project_DA/blob/main/churn_bysrcitizen.png)

```python
#Do senior citizen churn more?using plot
sns.countplot(x='SeniorCitizen', hue='Churn', data=df1)
plt.title('Churn by Senior Citizen Status')
plt.xlabel('Senior Citizen (0 = No, 1 = Yes)')
plt.ylabel('Number of Customers')

plt.show()
```
![An Image](https://github.com/Dataprofessional2/Customer_Churn_Project_DA/blob/main/bar_srcitizen_churn.png)

```python
# Churn by tenure
df1.groupby('Churn')['tenure'].describe()
```
![An Image](https://github.com/Dataprofessional2/Customer_Churn_Project_DA/blob/main/tenure_churn.png)

```python
# Churn by tenure boxplot
sns.boxplot(x='Churn', y='tenure', data=df1)
plt.title('Tenure Distribution by Churn')
plt.xlabel('Churn')
plt.ylabel('Tenure (Months)')

plt.show()
```
![An Image](https://github.com/Dataprofessional2/Customer_Churn_Project_DA/blob/main/tenure_boxplot.png)

```python
# Churn count by contract type
df1.groupby('Contract')['Churn'].value_counts()
```
![An Image](https://github.com/Dataprofessional2/Customer_Churn_Project_DA/blob/main/ct_churn.png)
```python
# Churn count by contract type
sns.countplot(x='Contract', hue='Churn', data=df1)

plt.title('Churn by Contract Type')
plt.xlabel('Contract Type')
plt.ylabel('Number of Customers')

plt.show()
```
![An Image](https://github.com/Dataprofessional2/Customer_Churn_Project_DA/blob/main/ct_bar.png)


```python
# Churn count by payment method
df1.groupby('PaymentMethod')['Churn'].value_counts()
```
![An Image](https://github.com/Dataprofessional2/Customer_Churn_Project_DA/blob/main/pm_churn.png)

```python
sns.countplot(x='PaymentMethod', hue='Churn', data=df1)

plt.title('Churn by Payment Method')
plt.xlabel('Payment Method')
plt.ylabel('Number of Customers')

plt.xticks(rotation=45)

plt.show()
```
![An Image](https://github.com/Dataprofessional2/Customer_Churn_Project_DA/blob/main/pm_bar.png)

```python
# Churn count by internet service
df1.groupby('InternetService')['Churn'].value_counts()
```
![An Image](https://github.com/Dataprofessional2/Customer_Churn_Project_DA/blob/main/churn_is.png)
```python
sns.countplot(x='InternetService', hue='Churn', data=df1)

plt.title('Churn by Internet Service')
plt.xlabel('Internet Service Type')
plt.ylabel('Number of Customers')

plt.show()
```
![An Image](https://github.com/Dataprofessional2/Customer_Churn_Project_DA/blob/main/is_bar.png)

```python
df1.groupby('TechSupport')['Churn'].value_counts()
```
![An Image](https://github.com/Dataprofessional2/Customer_Churn_Project_DA/blob/main/ts_churn.png)

```python
sns.countplot(x='TechSupport', hue='Churn', data=df1)

plt.title('Churn by Tech Support')
plt.xlabel('Tech Support')
plt.ylabel('Number of Customers')

plt.show()
```
![An Image](https://github.com/Dataprofessional2/Customer_Churn_Project_DA/blob/main/ts_bar.png)

## Feature Engineering
```python
#Feature Engineering
# Make a copy to avoid modifying original data
df_model = df1.copy()


# Fix TotalCharges (common issue in Telco dataset)

df_model['TotalCharges'] = pd.to_numeric(
    df_model['TotalCharges'],
    errors='coerce'
)


# Handle missing values

df_model['TotalCharges'].fillna(
    df_model['TotalCharges'].median(),
    inplace=True
)


# Clean target variable

df_model['Churn'] = df_model['Churn'].str.strip()

df_model['Churn'] = df_model['Churn'].map({
    'No': 0,
    'Yes': 1
})


# Create tenure groups

df_model['tenure_group'] = pd.cut(
    df_model['tenure'],
    bins=[0, 12, 24, 48, 72],
    labels=[
        '0-1 Year',
        '1-2 Years',
        '2-4 Years',
        '4-6 Years'
    ]
)


# Create MonthlyCharges group

df_model['MonthlyCharges_group'] = pd.cut(
    df_model['MonthlyCharges'],
    bins=[0, 35, 70, 120],
    labels=[
        'Low',
        'Medium',
        'High'
    ]
)


# One-hot encoding

df_encoded = pd.get_dummies(
    df_model,
    drop_first=True
)


# Scale numeric variables

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

num_cols = [
    'tenure',
    'MonthlyCharges',
    'TotalCharges'
]

df_encoded[num_cols] = scaler.fit_transform(
    df_encoded[num_cols]
)


# Verify target

df_encoded['Churn'].value_counts()
```
![An Image](https://github.com/Dataprofessional2/Customer_Churn_Project_DA/blob/main/feature%20engineering.png)


## Handling Class Imbalance
```python
# Separate features and target
X = df_encoded.drop('Churn', axis=1)
y = df_encoded['Churn']

# Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Apply SMOTE
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(
    X_train,
    y_train
)


# Check balance

y_train_smote.value_counts()
```
![An Image](https://github.com/Dataprofessional2/Customer_Churn_Project_DA/blob/main/class_imbalance.png)



```python
#Building BaseLine Logistic Regression
from sklearn.linear_model import LogisticRegression

# Create model
log_model = LogisticRegression(max_iter=1000)

# Train model
log_model.fit(
    X_train_smote,
    y_train_smote
)

# Predict on test data

y_pred = log_model.predict(X_test)

# Predict probabilities (needed for ROC-AUC)

y_prob = log_model.predict_proba(X_test)[:, 1]
```

![An Image](https://github.com/Dataprofessional2/Customer_Churn_Project_DA/blob/main/baseline%20logistic%20regression.png)

```python
#Logistic Regression Evaluation
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
```
![An Image](https://github.com/Dataprofessional2/Customer_Churn_Project_DA/blob/main/Log_regression.png)

```python
#Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
import seaborn as sns
import matplotlib.pyplot as plt
sns.heatmap(
    cm,
    annot=True,
    fmt='d')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
```
![An Image](https://github.com/Dataprofessional2/Customer_Churn_Project_DA/blob/main/confusion_matrix.png)
![An Image](https://github.com/Dataprofessional2/Customer_Churn_Project_DA/blob/main/confusion_matrix_viz.png)


```python
#Tradeoff between TPR AND FPR
from sklearn.metrics import roc_auc_score

roc_auc = roc_auc_score(y_test, y_prob)

print("ROC-AUC Score:", roc_auc)

from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))
```
![An Image](https://github.com/Dataprofessional2/Customer_Churn_Project_DA/blob/main/ROC.png)





```python
#Identifying High risk consumer
# Create results table

results = X_test.copy()

results['Actual_Churn'] = y_test.values

results['Churn_Probability'] = y_prob


# Predict churn using threshold

results['Predicted_Churn'] = (
    results['Churn_Probability'] >= 0.5
).astype(int)


# Show customers likely to churn

high_risk_customers = results[
    results['Predicted_Churn'] == 1
]

high_risk_customers.head()
```
![An Image](https://github.com/Dataprofessional2/Customer_Churn_Project_DA/blob/main/high_risk_consumer.png)


## Building Random Forest Model
```python
#Building Random Forest Model
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
rf_model.fit(
    X_train_smote,
    y_train_smote
)
#Make predictions
y_pred_rf = rf_model.predict(X_test)
y_prob_rf = rf_model.predict_proba(X_test)[:, 1]
# Evaluate Model
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Precision:", precision_score(y_test, y_pred_rf))
print("Recall:", recall_score(y_test, y_pred_rf))
print("F1 Score:", f1_score(y_test, y_pred_rf))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred_rf))
```
![An Image](https://github.com/Dataprofessional2/Customer_Churn_Project_DA/blob/main/build%20rf%20model.png)
![An Image](https://github.com/Dataprofessional2/Customer_Churn_Project_DA/blob/main/Screenshot%202026-04-23%20145109.png)





```python
from sklearn.metrics import accuracy_score

# Training accuracy
train_pred = log_model.predict(X_train_smote)

print("Train Accuracy:",
      accuracy_score(y_train_smote, train_pred))

# Test accuracy
print("Test Accuracy:",
      accuracy_score(y_test, y_pred))
import pandas as pd

feature_importance_rf = pd.DataFrame({
    'Feature': X_train_smote.columns,
    'Importance': rf_model.feature_importances_
})

feature_importance_rf = feature_importance_rf.sort_values(
    by='Importance',
    ascending=False
)

feature_importance_rf.head(10)
```
![An Image](https://github.com/Dataprofessional2/Customer_Churn_Project_DA/blob/main/perfomance_ondata.png)
![An Image](https://github.com/Dataprofessional2/Customer_Churn_Project_DA/blob/main/feat_importance.png)



## Customer Churn Dashboard
![An Image](https://github.com/Dataprofessional2/Customer_Churn_Project_DA/blob/main/Dashboard.png)












