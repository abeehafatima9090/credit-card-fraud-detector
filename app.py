import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Sklearn and related libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
from sklearn.metrics import (confusion_matrix, precision_score, recall_score,
                           roc_auc_score, accuracy_score, f1_score, precision_recall_curve)

# XGBoost
import xgboost as xgb

# Suppress warnings for clarity
import warnings
warnings.filterwarnings("ignore")

# Set a random seed for reproducibility
RANDOM_SEED = 42

def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    """Custom function to plot confusion matrix using matplotlib"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

# Load and prepare data
data_path = '/content/creditcard.csv'
df = pd.read_csv(data_path)
print("Initial dataset shape:", df.shape)

# Quick glance at the data
print("Sample data:")
display(df.head())
print("\nData Info:")
df.info()

# Check for missing values (percentage)
missing_percentage = df.isnull().mean() * 100
print("\nMissing values (in %):")
print(missing_percentage)

# Target distribution visualization
print("\nTarget Distribution (Counts):")
print(df['Class'].value_counts())
print("\nTarget Distribution (Proportion):")
print(df['Class'].value_counts(normalize=True))

plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='Class')
plt.title("Class Distribution")
plt.xlabel("Class")
plt.ylabel("Count")
plt.show()

# Data preprocessing
cols_to_drop = missing_percentage[missing_percentage > 50].index
print(f"Dropping columns with >50% missing values: {list(cols_to_drop)}")
df.drop(columns=cols_to_drop, inplace=True)

# Handle categorical variables
categorical_columns = df.select_dtypes(include=['object']).columns
print("Categorical columns found:", list(categorical_columns))
df = pd.get_dummies(df, columns=categorical_columns)

# Split features and target
X = df.drop(columns=['Class'])
y = df['Class']

# Initial train-test split (80-20)
X_train_total, X_test, y_train_total, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_SEED, shuffle=True, stratify=y
)

# Split train into train and validation (70-30)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_total, y_train_total, test_size=0.3, random_state=RANDOM_SEED,
    shuffle=True, stratify=y_train_total
)

print("Train set shape:", X_train.shape)
print("Validation set shape:", X_val.shape)
print("Test set shape:", X_test.shape)

# Calculate scale_pos_weight
n_fraud = sum(y_train)
n_legit = len(y_train) - n_fraud
scale_pos_weight_value = n_legit / n_fraud if n_fraud != 0 else 1
print("scale_pos_weight_value:", scale_pos_weight_value)

# Model 1: XGBoost with class weight
model_XGB_us = xgb.XGBClassifier(
    objective='binary:logistic',
    learning_rate=0.05,
    max_depth=6,
    n_estimators=100,
    scale_pos_weight=1500,
    use_label_encoder=False,
    eval_metric='logloss'
)

# Normalize features
scaler = Normalizer()
X_train_sc = scaler.fit_transform(X_train)
X_val_sc = scaler.transform(X_val)

# Train Model 1
model_XGB_us.fit(X_train_sc, y_train)
y_pred_val_model1 = model_XGB_us.predict(X_val_sc)

# Plot confusion matrix for Model 1
plot_confusion_matrix(y_val, y_pred_val_model1, "Confusion Matrix - Model 1 (Validation)")

# Print metrics for Model 1
print("Model 1 Validation Metrics:")
print("Accuracy:", accuracy_score(y_val, y_pred_val_model1))
print("Precision:", precision_score(y_val, y_pred_val_model1))
print("Recall:", recall_score(y_val, y_pred_val_model1))
print("F1 Score:", f1_score(y_val, y_pred_val_model1))

# Retrain on full training data
X_train_total_sc = scaler.fit_transform(X_train_total)
model_XGB_us.fit(X_train_total_sc, y_train_total)
y_pred_train_total = model_XGB_us.predict(X_train_total_sc)

# Create filtered dataset
df_studio_us = X_train_total.copy()
df_studio_us['Class'] = y_train_total
df_studio_us['y_pred'] = y_pred_train_total

# Filter out correctly predicted non-fraud cases
filter_mask = ~((df_studio_us['Class'] == 0) & (df_studio_us['Class'] == df_studio_us['y_pred']))
df_studio_us = df_studio_us[filter_mask]

print("New dataset shape after filtering:", df_studio_us.shape)
print("New dataset distribution:")
print(df_studio_us['Class'].value_counts())

plt.figure(figsize=(8, 6))
sns.countplot(data=df_studio_us, x='Class')
plt.title("Class Distribution After Filtering")
plt.xlabel("Class")
plt.ylabel("Count")
plt.show()

# Prepare filtered dataset
dataset_us_X = df_studio_us.drop(columns=['Class','y_pred'])
dataset_us_y = df_studio_us['Class']

# Split filtered data
X_train_us, X_test_us, y_train_us, y_test_us = train_test_split(
    dataset_us_X, dataset_us_y, test_size=0.2, random_state=RANDOM_SEED,
    shuffle=True, stratify=dataset_us_y
)

print("Filtered Train set shape:", X_train_us.shape)
print("Filtered Test set shape:", X_test_us.shape)

# Model 2: XGBoost without class weight
model_XGB_2 = xgb.XGBClassifier(
    objective='binary:logistic',
    learning_rate=0.05,
    max_depth=6,
    n_estimators=100,
    use_label_encoder=False,
    eval_metric='logloss'
)

# Normalize features for Model 2
scaler_2 = Normalizer()
X_train_us_sc = scaler_2.fit_transform(X_train_us)
X_test_us_sc = scaler_2.transform(X_test_us)

# Train Model 2
model_XGB_2.fit(X_train_us_sc, y_train_us)
y_pred_test_us = model_XGB_2.predict(X_test_us_sc)

# Plot confusion matrix for Model 2
plot_confusion_matrix(y_test_us, y_pred_test_us, "Confusion Matrix - Model 2 (Filtered Test)")

# Print metrics for Model 2
print("Model 2 (Filtered Test) Metrics:")
print("Accuracy:", accuracy_score(y_test_us, y_pred_test_us))
print("Precision:", precision_score(y_test_us, y_pred_test_us))
print("Recall:", recall_score(y_test_us, y_pred_test_us))
print("F1 Score:", f1_score(y_test_us, y_pred_test_us))

y_prob_test_us = model_XGB_2.predict_proba(X_test_us_sc)[:, 1]
print("AUC:", roc_auc_score(y_test_us, y_prob_test_us))

# Final ensemble evaluation
X_test_sc = scaler.transform(X_test)
y_test_pred_model1 = model_XGB_us.predict(X_test_sc)

# Get predictions for samples flagged as fraud by Model 1
X_test_for_model2 = X_test[y_test_pred_model1 == 1]
y_test_for_model2 = y_test[y_test_pred_model1 == 1]

X_test_for_model2_sc = scaler_2.transform(X_test_for_model2)
y_test_pred_model2 = model_XGB_2.predict(X_test_for_model2_sc)

# Combine predictions
final_y_test_pred = y_test_pred_model1.copy()
final_y_test_pred[y_test_pred_model1 == 1] = y_test_pred_model2

# Print final metrics
print("Final Evaluation on Unseen Test Data:")
print("Accuracy:", accuracy_score(y_test, final_y_test_pred))
print("Precision:", precision_score(y_test, final_y_test_pred))
print("Recall:", recall_score(y_test, final_y_test_pred))
print("F1 Score:", f1_score(y_test, final_y_test_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, final_y_test_pred))

# Plot final confusion matrix
plot_confusion_matrix(y_test, final_y_test_pred, "Confusion Matrix - Final Ensemble")
import pickle

def save_models(model1, model2, scaler1, scaler2):
    """Saves models and scalers to disk using pickle."""
    # Save Model 1
    with open('model1.pkl', 'wb') as f:
        pickle.dump(model1, f)

    # Save Model 2
    with open('model2.pkl', 'wb') as f:
        pickle.dump(model2, f)

    # Save Scaler 1
    with open('scaler1.pkl', 'wb') as f:
        pickle.dump(scaler1, f)

    # Save Scaler 2
    with open('scaler2.pkl', 'wb') as f:
        pickle.dump(scaler2, f)

    print("Models and scalers saved successfully!")

    # You can return a value if needed, or simply call the function
    # without assigning it to a variable

# Now you can call the function:
new_var = save_models(model_XGB_us, model_XGB_2, scaler, scaler_2)
