# Customer Churn Prediction

This project aims to predict customer churn based on behavioral and demographic data, offering business insights into customer retention strategies.

##  Dataset Overview

- **Rows**: 3,150 (after removing 300 duplicates: 2,850 remain)
- **Features**: 14 numerical columns
- **Target Variable**: `Churn` (Binary: 0 = No, 1 = Yes)

### Feature Groups:
- **Usage**: Seconds of Use, Frequency of Use, Frequency of SMS, Distinct Called Numbers
- **Billing/Plan**: Charge Amount, Tariff Plan, Subscription Length
- **Customer Info**: Age, Age Group, Customer Value, Status, Complains, Call Failure

##  Data Processing

- Checked for and removed missing values and duplicates.
- Applied SMOTE for class imbalance (~15.6% churn rate).
- Scaled numerical features and analyzed distribution for skewness.
- Identified multicollinearity among features.

##  Exploratory Data Analysis

- **Complains** and **Status** strongly correlate with churn.
- **Customer Value** and **Usage metrics** negatively correlate with churn.
- Plans and statuses give actionable insights into customer satisfaction.

##  Sample Visualizations

These are a few key charts from the analysis:

![Churn Distribution](visualizations/churn_distribution.png)
![Correlation Heatmap](visualizations/correlation_heatmap.png)
![Feature Importance - SHAP](visualizations/shap_feature_importance.png)

##  Models Evaluated

| Model                  | Accuracy | Precision (Churn=1) | Recall (Churn=1) | F1-Score (Churn=1) |
|------------------------|----------|----------------------|-------------------|---------------------|
| Logistic Regression    | 83%      | 0.48                 | 0.88              | 0.62                |
| K-Nearest Neighbors    | 92%      | 0.67                 | 0.93              | 0.78                |
| Support Vector Machine | 88%      | 0.58                 | 0.95              | 0.72                |
| Random Forest (Untuned)| 96%      | 0.83                 | 0.93              | 0.88                |
| XGBoost (Untuned)      | 94%      | 0.79                 | 0.88              | 0.83                |
| Random Forest (Tuned)  | 96%      | 0.83                 | 0.93              | 0.88                |
| XGBoost (Tuned)        | 96%      | 0.81                 | 0.94              | 0.87                |

##  Feature Importance & Interpretation

- **Top Features** (SHAP): Complains, Frequency of SMS, Seconds of Use, Customer Value
- Tree-based models like Random Forest & XGBoost handled multicollinearity better.
- SHAP and permutation importance used to interpret predictions.

##  Key Takeaways

- **Best Overall Model**: Tuned XGBoost (balanced performance on precision, recall, F1)
- **High Recall Models**: SVM and Logistic Regression (good for minimizing missed churns)
- **Best Trade-off**: Random Forest (Tuned)


 For code, preprocessing, visualizations, and evaluation details, see the notebook/script files and plots in this repository.
