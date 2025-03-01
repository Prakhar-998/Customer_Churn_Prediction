# Customer Churn Prediction

## Overview
This project analyzes customer churn using a dataset containing various customer attributes such as age, tenure, subscription type, and total spend. The goal is to explore factors influencing churn and develop insights using data preprocessing and visualization techniques.

## Dataset
The dataset consists of **440,833** rows and **12** columns. Key features include:
- `Age` (int)
- `Gender` (categorical: Male/Female, later converted to numerical values)
- `Tenure` (float)
- `Usage Frequency` (float)
- `Support Calls` (float)
- `Payment Delay` (float)
- `Subscription Type` (categorical: Basic/Standard/Premium)
- `Contract Length` (categorical: Monthly/Quarterly/Annual)
- `Total Spend` (float)
- `Last Interaction` (float)
- `Churn` (binary: 0 = No, 1 = Yes)

## Project Workflow
### 1. Data Loading
The dataset is loaded from Google Drive into a Pandas DataFrame.
```python
from google.colab import drive
import pandas as pd

drive.mount('/content/drive')
dataset = pd.read_csv("/content/drive/MyDrive/Customer_churn_dataset/customer_churn_dataset-training-master.csv")
```

### 2. Data Cleaning & Preprocessing
- Removed `CustomerID` as it does not contribute to analysis.
- Converted `Age` to a numeric type and handled missing values.
- Dropped rows with missing values to maintain dataset integrity.
- Encoded categorical variables (`Gender`, `Subscription Type`, `Contract Length`) for numerical processing.

### 3. Data Visualization
- Churn analysis based on tenure:
```python
import matplotlib.pyplot as plt
plt.hist([dataset[dataset.Churn==1]['Tenure'], dataset[dataset.Churn==0]['Tenure']],
         color=['green','red'], label=['Churn = Yes', 'Churn = No'])
plt.xlabel("Tenure")
plt.ylabel("Number of Customers")
plt.title("Customer Churn Prediction Analysis")
plt.legend()
plt.show()
```
- Churn analysis based on total spend.

### 4. Encoding & Feature Engineering
- Converted `Gender` to binary (Female=1, Male=0)
- Applied one-hot encoding on categorical features (`Subscription Type`, `Contract Length`)

### 5. Dataset Finalization
A final dataset with preprocessed and numerical values is prepared for further modeling.
```python
df = pd.get_dummies(data=dataset, columns=["Subscription Type", "Contract Length"], dtype=int)
```

## Requirements
To run this project, install the necessary dependencies:
```sh
pip install pandas matplotlib
```

## Usage
1. Open the notebook in Google Colab.
2. Mount Google Drive and load the dataset.
3. Run the preprocessing steps to clean and transform data.
4. Visualize churn-related insights.

## Future Scope
- Train and evaluate machine learning models for churn prediction.
- Implement feature selection and hyperparameter tuning for better accuracy.
- Deploy a predictive model as a web application.

## Author
[Prakhar Jaiswal]((https://github.com/Prakhar-998))

