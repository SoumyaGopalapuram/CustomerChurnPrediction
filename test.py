import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("Dataset Gopalapuram.csv")  # Replace with actual filename

print("Shape:", df.shape)
print("First 5 rows:\n", df.head())


missing = df.isnull().sum()
missing = missing[missing > 0]
if not missing.empty:
    print("\nMissing values found:\n", missing)
else:
    print("\n No missing values found.")


duplicates = df.duplicated().sum()
print(f"\nDuplicate rows: {duplicates}")
if duplicates > 0:
    df.drop_duplicates(inplace=True)
    print("Duplicates removed. New shape:", df.shape)

print("\nData types:\n", df.dtypes)
pd.set_option('display.max_columns', None)  # show all columns
pd.set_option('display.width', None)        # allow wide output
pd.set_option('display.max_rows', None)     # show all rows (for large describe)

print("\nSummary statistics:\n", df.describe())

import matplotlib.pyplot as plt
import seaborn as sns

#Churn Distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='Churn', data=df, palette='pastel')
plt.title("Churn Distribution")
plt.xlabel("Churn (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

#Percentage
churn_rate = df['Churn'].value_counts(normalize=True) * 100
print("\nChurn Rate:\n", churn_rate.round(2))

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

#Split features and target
X = df.drop(columns=['Churn'])
y = df['Churn']

#Train-test split with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

#Apply SMOTE to training data only
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

#Check balance
print("Original class distribution:\n", y_train.value_counts())
print("\nAfter SMOTE:\n", y_train_bal.value_counts())

#Standardize features (important for LogReg, SVM, KNN)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_bal)
X_test_scaled = scaler.transform(X_test)

# Combine balanced arrays back into a DataFrame
X_train_bal_df = pd.DataFrame(X_train_bal, columns=X.columns)
y_train_bal_df = pd.Series(y_train_bal, name="Churn")
balanced_df = pd.concat([X_train_bal_df, y_train_bal_df], axis=1)

# Convert key categorical variables (if needed)
categorical_vars = ['Complains', 'Tariff Plan', 'Status']

# Plot each categorical variable vs balanced Churn
import seaborn as sns
import matplotlib.pyplot as plt

for var in categorical_vars:
    plt.figure(figsize=(6, 4))
    sns.countplot(data=balanced_df, x=var, hue='Churn', palette='pastel')
    plt.title(f"[Balanced] Churn by {var}")
    plt.xlabel(var)
    plt.ylabel("Count")
    plt.legend(title='Churn', labels=['No', 'Yes'])
    plt.tight_layout()
    plt.show()

import seaborn as sns
import matplotlib.pyplot as plt

#Correlation matrix with balanced data
corr_bal = balanced_df.corr()

#Plot
plt.figure(figsize=(12, 10))
sns.heatmap(corr_bal, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title("Correlation Matrix")
plt.tight_layout()
plt.show()


#List of numerical features (excluding target)
numerical_vars = df.drop(columns=['Churn']).select_dtypes(include=['int64', 'float64']).columns.tolist()

# Plot histograms
import matplotlib.pyplot as plt

df[numerical_vars].hist(figsize=(14, 10), bins=20, edgecolor='black', color='lightblue')
plt.suptitle("Distribution of Numerical Features", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

#Apply log1p transformation to skewed features
from sklearn.preprocessing import StandardScaler
import numpy as np

#Copy data
X = df.drop(columns=['Churn']).copy()
y = df['Churn']

#Log-transform highly skewed features
skewed_features = ['Customer Value', 'Seconds of Use', 'Frequency of use', 'Frequency of SMS', 'Charge  Amount']
for feature in skewed_features:
    X[feature] = np.log1p(X[feature])  # log1p avoids log(0)

#Train-test split (before SMOTE!)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)

#Apply SMOTE only to training data
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

#Scale all features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_sm)
X_test_scaled = scaler.transform(X_test)



from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix


def evaluate_model(model, name):
    model.fit(X_train_scaled, y_train_sm)
    preds = model.predict(X_test_scaled)
    print(f"\n{name} Classification Report:\n", classification_report(y_test, preds))
    print(f"{name} Confusion Matrix:\n", confusion_matrix(y_test, preds))

# Logistic Regression
log_model = LogisticRegression(max_iter=1000, random_state=42)
evaluate_model(log_model, "Logistic Regression")

# K-Nearest Neighbors
knn_model = KNeighborsClassifier(n_neighbors=5)
evaluate_model(knn_model, "K-Nearest Neighbors")

# Support Vector Machine
svm_model = SVC(kernel='rbf', probability=True, random_state=42)
evaluate_model(svm_model, "Support Vector Machine")

# Random Forest (Untuned)
rf = RandomForestClassifier(random_state=42)
evaluate_model(rf, "Random Forest (Untuned)")

# XGBoost (Untuned)
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
evaluate_model(xgb, "XGBoost (Untuned)")

# Random Forest (Tuned)
rf_params = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}
grid_rf = GridSearchCV(rf, rf_params, cv=3, scoring='f1', verbose=0)
grid_rf.fit(X_train_scaled, y_train_sm)
print("Best RF Parameters:", grid_rf.best_params_)
evaluate_model(grid_rf.best_estimator_, "Random Forest (Tuned)")

# XGBoost (Tuned)
xgb_params = {
    'learning_rate': [0.01, 0.1],
    'max_depth': [5, 10],
    'n_estimators': [100, 200],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}
grid_xgb = GridSearchCV(xgb, xgb_params, cv=3, scoring='f1', verbose=0)
grid_xgb.fit(X_train_scaled, y_train_sm)
print("Best XGB Parameters:", grid_xgb.best_params_)
evaluate_model(grid_xgb.best_estimator_, "XGBoost (Tuned)")


#Feature Importance
import shap
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.inspection import permutation_importance


print("\n🔹 Logistic Regression Coefficients")
lr_importance = pd.Series(log_model.coef_[0], index=X.columns)
lr_importance.sort_values().plot(kind='barh', title="Logistic Regression Feature Importance")
plt.tight_layout()
plt.show()


print("\n🔹 SVM Permutation Importance")
svm_perm = permutation_importance(svm_model, X_test_scaled, y_test, n_repeats=10, random_state=42)
svm_importance = pd.Series(svm_perm.importances_mean, index=X.columns)
svm_importance.sort_values().plot(kind='barh', title="SVM Permutation Importance")
plt.tight_layout()
plt.show()


print("\n🔹 KNN Permutation Importance")
knn_perm = permutation_importance(knn_model, X_test_scaled, y_test, n_repeats=10, random_state=42)
knn_importance = pd.Series(knn_perm.importances_mean, index=X.columns)
knn_importance.sort_values().plot(kind='barh', title="KNN Permutation Importance")
plt.tight_layout()
plt.show()


print("\n🔹 Random Forest (Untuned) SHAP")
shap_values = shap.TreeExplainer(rf).shap_values(X)
shap.summary_plot(shap_values, X, plot_type="bar")
plt.title("Random Forest (Untuned) SHAP")
plt.tight_layout()
plt.show()


print("\n🔹 Random Forest SHAP")
shap_values = shap.TreeExplainer(grid_rf.best_estimator_).shap_values(X)
shap.summary_plot(shap_values, X, plot_type="bar")
plt.title("Random Forest (Tuned) SHAP")
plt.tight_layout()
plt.show()


print("\n🔹 XGBoost (Untuned) SHAP")
xgb_explainer = shap.Explainer(xgb)
xgb_shap_values = xgb_explainer(X_test)
shap.plots.beeswarm(xgb_shap_values, max_display=15)
plt.title("XGBoost (Untuned) SHAP")
plt.tight_layout()
plt.show()


print("\n🔹 XGBoost SHAP")
xgb_explainer = shap.Explainer(grid_xgb.best_estimator_)
xgb_shap_values = xgb_explainer(X_test)
shap.plots.beeswarm(xgb_shap_values, max_display=15)
plt.title("XGBoost (Tuned) SHAP")
plt.tight_layout()
plt.show()


