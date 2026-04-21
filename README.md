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
- Advanced Models: Random Forest and XGBoost
- Model Evaluation
- Performance Metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- Insights and Recommendations
- Customer Retention Strategies
- Identifying High-Risk Customers
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

The primary objective of this project is to predict whether a customer is likely to churn based on features such as demographic information, service usage, contract type, and billing behavior. By identifying patterns and relationships within the data, we aim to help businesses detect high-risk customers early and implement effective retention strategies to reduce customer loss and improve overall customer satisfaction.


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
```
![An Image](https://github.com/Dataprofessional2/Customer_Churn_Project_DA/blob/main/monthly_price_churn.png)
```python

```
```python

````python

``````python

``````python

``````python

``````python

``````python

``````python

`````
