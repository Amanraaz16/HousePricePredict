import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import joblib
import os

# Load the data
print("Loading data...")
df = pd.read_csv('Housing.csv')

# Display basic info
print(f"Dataset shape: {df.shape}")
print(f"\nFirst few rows:")
print(df.head())
print(f"\nData types:")
print(df.dtypes)
print(f"\nMissing values:")
print(df.isnull().sum())

# Separate features and target
X = df.drop('price', axis=1)
y = df['price']

# Preprocessing
print("\nPreprocessing data...")

# Encode binary categorical variables (yes/no) to 1/0
binary_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
for col in binary_cols:
    X[col] = X[col].map({'yes': 1, 'no': 0})

# Encode furnishingstatus using label encoding
le = LabelEncoder()
X['furnishingstatus'] = le.fit_transform(X['furnishingstatus'])

# Save the label encoder for later use
joblib.dump(le, 'furnishing_encoder.pkl')

print("\nPreprocessed data:")
print(X.head())
print(f"\nFeature columns: {X.columns.tolist()}")

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler
joblib.dump(scaler, 'scaler.pkl')

print("\nTraining multiple models...")

# Define models to try
models = {
    'Random Forest': RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42),
    'XGBoost': xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42),
    'Ridge': Ridge(alpha=1.0),
    'Lasso': Lasso(alpha=1.0)
}

# Train and evaluate models
results = {}
best_model = None
best_score = float('inf')
best_model_name = None

for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # Use scaled features for linear models, original for tree-based models
    if name in ['Ridge', 'Lasso']:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        # Cross-validation with scaled features
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='neg_mean_squared_error')
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        # Cross-validation with original features
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    cv_rmse = np.sqrt(-cv_scores.mean())
    
    results[name] = {
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'CV_RMSE': cv_rmse
    }
    
    print(f"  RMSE: {rmse:,.2f}")
    print(f"  MAE: {mae:,.2f}")
    print(f"  R² Score: {r2:.4f}")
    print(f"  CV RMSE: {cv_rmse:,.2f}")
    
    # Track best model based on CV RMSE
    if cv_rmse < best_score:
        best_score = cv_rmse
        best_model = model
        best_model_name = name

print("\n" + "="*60)
print("MODEL COMPARISON")
print("="*60)
for name, metrics in results.items():
    print(f"\n{name}:")
    print(f"  RMSE: {metrics['RMSE']:,.2f}")
    print(f"  MAE: {metrics['MAE']:,.2f}")
    print(f"  R² Score: {metrics['R2']:.4f}")
    print(f"  CV RMSE: {metrics['CV_RMSE']:,.2f}")

print(f"\n{'='*60}")
print(f"Best Model: {best_model_name}")
print(f"Best CV RMSE: {best_score:,.2f}")
print(f"{'='*60}")

# Hyperparameter tuning for the best model
print(f"\nPerforming hyperparameter tuning for {best_model_name}...")

if best_model_name == 'Random Forest':
    param_grid = {
        'n_estimators': [200, 300, 400],
        'max_depth': [15, 20, 25],
        'min_samples_split': [2, 5, 10]
    }
    base_model = RandomForestRegressor(random_state=42, n_jobs=-1)
    grid_search = GridSearchCV(base_model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    print(f"Best parameters: {grid_search.best_params_}")
    
elif best_model_name == 'XGBoost':
    param_grid = {
        'n_estimators': [200, 300],
        'max_depth': [5, 6, 7],
        'learning_rate': [0.05, 0.1, 0.15]
    }
    base_model = xgb.XGBRegressor(random_state=42)
    grid_search = GridSearchCV(base_model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    print(f"Best parameters: {grid_search.best_params_}")
    
elif best_model_name == 'Gradient Boosting':
    param_grid = {
        'n_estimators': [200, 300],
        'max_depth': [4, 5, 6],
        'learning_rate': [0.05, 0.1, 0.15]
    }
    base_model = GradientBoostingRegressor(random_state=42)
    grid_search = GridSearchCV(base_model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    print(f"Best parameters: {grid_search.best_params_}")
    
elif best_model_name == 'Ridge':
    param_grid = {
        'alpha': [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
    }
    base_model = Ridge()
    grid_search = GridSearchCV(base_model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)
    grid_search.fit(X_train_scaled, y_train)
    best_model = grid_search.best_estimator_
    print(f"Best parameters: {grid_search.best_params_}")
    
elif best_model_name == 'Lasso':
    param_grid = {
        'alpha': [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
    }
    base_model = Lasso()
    grid_search = GridSearchCV(base_model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)
    grid_search.fit(X_train_scaled, y_train)
    best_model = grid_search.best_estimator_
    print(f"Best parameters: {grid_search.best_params_}")

# Final evaluation of tuned model
if best_model_name in ['Ridge', 'Lasso']:
    y_pred_final = best_model.predict(X_test_scaled)
else:
    y_pred_final = best_model.predict(X_test)

final_rmse = np.sqrt(mean_squared_error(y_test, y_pred_final))
final_mae = mean_absolute_error(y_test, y_pred_final)
final_r2 = r2_score(y_test, y_pred_final)

print(f"\nFinal Model Performance:")
print(f"  RMSE: {final_rmse:,.2f}")
print(f"  MAE: {final_mae:,.2f}")
print(f"  R² Score: {final_r2:.4f}")

# Save the best model
if best_model_name in ['Ridge', 'Lasso']:
    # Save model that uses scaled features
    joblib.dump(best_model, 'model.pkl')
    # Save flag to indicate if scaling is needed
    joblib.dump(True, 'use_scaling.pkl')
else:
    # Save model that uses original features
    joblib.dump(best_model, 'model.pkl')
    joblib.dump(False, 'use_scaling.pkl')

print(f"\nModel saved as 'model.pkl'")
print(f"Scaler saved as 'scaler.pkl'")
print(f"Label encoder saved as 'furnishing_encoder.pkl'")
print("\nTraining completed successfully!")

# Save feature names for the UI
feature_names = X.columns.tolist()
joblib.dump(feature_names, 'feature_names.pkl')
print(f"Feature names saved: {feature_names}")

