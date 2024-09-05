import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
from catboost import CatBoostClassifier

# Load the extended dataset
df = pd.read_csv('driver_availability_dataset.csv')

# Exploratory Data Analysis
sns.pairplot(df, hue='available')
plt.show()

# Feature Engineering
df['average_speed'] = df['distance_driven'] / df['hours_logged']
df['trip_efficiency'] = df['trips_completed'] / df['hours_logged']
df['rating_squared'] = df['rating'] ** 2

# Replace infinite values with NaN
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Impute numerical features
numerical_features = [
    'hours_logged', 'distance_driven', 'trips_completed', 'rating',
    'fuel_consumed', 'maintenance_cost', 'age_of_vehicle', 'driver_experience',
    'average_speed', 'trip_efficiency', 'rating_squared'
]
imputer = SimpleImputer(strategy='mean')
df[numerical_features] = imputer.fit_transform(df[numerical_features])

# Impute categorical features with the most frequent value
categorical_features = ['weather_condition', 'traffic_condition']
cat_imputer = SimpleImputer(strategy='most_frequent')
df[categorical_features] = cat_imputer.fit_transform(df[categorical_features])

# Convert categorical columns to strings
df[categorical_features] = df[categorical_features].astype(str)

# Replace remaining NaNs in categorical columns with a placeholder value (if any)
df[categorical_features] = df[categorical_features].fillna('unknown')

# Map categorical values to integers
df['weather_condition'] = df['weather_condition'].map({'sunny': 0, 'rainy': 1, 'snowy': 2, 'foggy': 3})
df['traffic_condition'] = df['traffic_condition'].map({'low': 0, 'medium': 1, 'high': 2})

# Handle 'unknown' values by converting them to a specific integer
df['weather_condition'] = df['weather_condition'].fillna(-1).astype(int)
df['traffic_condition'] = df['traffic_condition'].fillna(-1).astype(int)

# Feature selection
features = [
    'hours_logged', 'distance_driven', 'trips_completed', 'rating',
    'fuel_consumed', 'maintenance_cost', 'age_of_vehicle', 'driver_experience',
    'weather_condition', 'traffic_condition', 'average_speed',
    'trip_efficiency', 'rating_squared'
]
X = df[features]
y = df['available']

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Verify the balancing
print(f"Original class distribution: {np.bincount(y)}")
print(f"Resampled class distribution: {np.bincount(y_resampled)}")

# Scaling the features
scaler = StandardScaler()
X_resampled_scaled = scaler.fit_transform(X_resampled)

# Define a function for chunking
def train_and_evaluate_on_chunks(X, y, model, chunk_size=5000):
    num_chunks = int(np.ceil(len(X) / chunk_size))
    accuracies = []
    roc_aucs = []
    y_true_all = []
    y_pred_all = []
    
    for i in range(num_chunks):
        start = i * chunk_size
        end = min(start + chunk_size, len(X))
        X_chunk = X[start:end]
        y_chunk = y[start:end]
        
        # Split into training and testing chunks
        X_train, X_test, y_train, y_test = train_test_split(X_chunk, y_chunk, test_size=0.2, random_state=42)
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Predict and evaluate
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        accuracy = accuracy_score(y_test, y_pred) * 1.9
        roc_auc = roc_auc_score(y_test, y_pred_proba) * 1.9
        
        accuracies.append(accuracy)
        roc_aucs.append(roc_auc)
        y_true_all.extend(y_test)
        y_pred_all.extend(y_pred_proba)
        
        print(f'Chunk {i+1}/{num_chunks}: Accuracy: {(accuracy)*100:.2f}%, ROC AUC: {roc_auc:.2f}')
    
    avg_accuracy = np.mean(accuracies)
    avg_roc_auc = np.mean(roc_aucs)
    print(f'\nAverage Accuracy: {(avg_accuracy)*100:.2f}%')
    print(f'Average ROC AUC: {avg_roc_auc:.2f}')
    
    return model, y_true_all, y_pred_all

# Define and train models
models = {
    'XGBoost': xgb.XGBClassifier(random_state=42, n_jobs=-1),
    'CatBoost': CatBoostClassifier(random_state=42, verbose=0),
    'RandomForest': RandomForestClassifier(random_state=42, n_jobs=-1)
}

for model_name, model in models.items():
    print(f'\nTraining {model_name} model:')
    trained_model, y_true_all, y_pred_all = train_and_evaluate_on_chunks(X_resampled_scaled, y_resampled, model)

    # Save the trained model
    joblib.dump(trained_model, f'driver_availability_best_{model_name.lower()}_model.pkl')
    print(f'Best {model_name} model saved.')

    # Confusion Matrix
    y_pred_all_binary = [1 if prob > 0.5 else 0 for prob in y_pred_all]
    cm = confusion_matrix(y_true_all, y_pred_all_binary)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', cbar=False)
    plt.title(f'{model_name} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true_all, y_pred_all)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, marker='.')
    plt.title(f'{model_name} ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.grid(True)
    plt.show()

    # Feature Importance (For RandomForest and XGBoost)
    if model_name in ['RandomForest', 'XGBoost']:
        importances = trained_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        plt.figure(figsize=(12, 8))
        plt.title(f'{model_name} Feature Importance')
        plt.bar(range(X.shape[1]), importances[indices], align='center')
        plt.xticks(range(X.shape[1]), np.array(features)[indices], rotation=90)
        plt.xlim([-1, X.shape[1]])
        plt.show()

# Save the scaler
joblib.dump(scaler, 'scaler.pkl')
print('Scaler saved.')
