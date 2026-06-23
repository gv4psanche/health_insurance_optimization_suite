import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_squared_error

# 1. Create Synthetic Portfolio Dataset
np.random.seed(42)
n_samples = 1000

data = pd.DataFrame({
    'customer_id': range(1001, 1001 + n_samples),
    'age': np.random.randint(18, 75, n_samples),
    'bmi': np.random.uniform(18.5, 40.0, n_samples),
    'pre_existing_conditions': np.random.randint(0, 4, n_samples),
    'monthly_premium': np.random.uniform(50, 300, n_samples),
    'failed_payments_3m': np.random.choice([0, 1, 2, 3], n_samples, p=[0.8, 0.12, 0.06, 0.02]),
    'customer_service_calls_6m': np.random.poisson(1.5, n_samples),
    'claim_amount_history': np.random.exponential(2000, n_samples),
    'recent_claim_distance_miles': np.random.exponential(15, n_samples)
})

# Generate target variables derived with controlled noise
data['is_fraud'] = ((data['claim_amount_history'] > 5000) & (data['recent_claim_distance_miles'] > 45)).astype(int)
data['has_chronic_disease'] = ((data['bmi'] > 30) & (data['age'] > 50)).astype(int)
data['next_month_claims_cost'] = (data['age'] * 5) + (data['pre_existing_conditions'] * 150) + np.random.normal(0, 50, n_samples)
data['will_churn'] = ((data['customer_service_calls_6m'] > 3) | (data['failed_payments_3m'] > 1)).astype(int)


# --------------------------------------------------------------------------------------
# 2. Financial Risk & Fraud Detection (Binary Classification)
# Business Goal: Flag potentially fraudulent claims based on high claim amounts submitted unusually far away from the customer's home address.
# Features and Target
X1 = data[['claim_amount_history', 'recent_claim_distance_miles', 'age']]
y1 = data['is_fraud']

X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size=0.2, random_state=42)

# Train XGBoost Classifier
fraud_model = xgb.XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42)
fraud_model.fit(X_train1, y_train1)

# Evaluate
preds1 = fraud_model.predict(X_test1)
print("--- Fraud Detection Results ---")
print(classification_report(y_test1, preds1))


# --------------------------------------------------------------------------------------
# 3. Customer Analytics: Churn & Propensity (Binary Classification)
# Business Goal: Identify at-risk policyholders likely to drop their insurance policy due to poor account health (failed payments) or friction (high customer service calls).
# Features and Target
X2 = data[['monthly_premium', 'failed_payments_3m', 'customer_service_calls_6m', 'age']]
y2 = data['will_churn']

X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.2, random_state=42)

# Train XGBoost Classifier
churn_model = xgb.XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42)
churn_model.fit(X_train2, y_train2)

# Evaluate
preds2 = churn_model.predict(X_test2)
print("--- Customer Churn Results ---")
print(classification_report(y_test2, preds2))


# --------------------------------------------------------------------------------------
# 4. Operational Forecasting (Regression)
# Business Goal: Forecast the financial payout amount the insurance company will owe next month across its book of business to ensure liquid capital reserves.
# Features and Target
X3 = data[['age', 'bmi', 'pre_existing_conditions', 'monthly_premium']]
y3 = data['next_month_claims_cost']

X_train3, X_test3, y_train3, y_test3 = train_test_split(X3, y3, test_size=0.2, random_state=42)

# Train XGBoost Regressor
forecast_model = xgb.XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42)
forecast_model.fit(X_train3, y_train3)

# Evaluate
preds3 = forecast_model.predict(X_test3)
rmse = np.sqrt(mean_squared_error(y_test3, preds3))
print("--- Operational Payout Forecasting Results ---")
print(f"Root Mean Squared Error (RMSE): ${rmse:.2f}")


# --------------------------------------------------------------------------------------
# 5. Operational Forecasting (Regression)
# Business Goal: Healthcare & Diagnostics (Risk Stratification)
# Features and Target
X4 = data[['age', 'bmi', 'pre_existing_conditions']]
y4 = data['has_chronic_disease']

X_train4, X_test4, y_train4, y_test4 = train_test_split(X4, y4, test_size=0.2, random_state=42)

# Train XGBoost Classifier
health_model = xgb.XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42)
health_model.fit(X_train4, y_train4)

# Evaluate
preds4 = health_model.predict(X_test4)
print("--- Healthcare Diagnostic Risk Results ---")
print(classification_report(y_test4, preds4))


# -------------------------------------------------------------------------------------
# Plot the Feature Importance charts
import matplotlib.pyplot as plt
import xgboost as xgb

# Set up a 2x2 grid plot for professional presentation
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('XGBoost Feature Importance Across Business Units', fontsize=16, fontweight='bold')

# 1. Financial Risk & Fraud Detection
xgb.plot_importance(fraud_model, ax=axes[0, 0], max_num_features=5, importance_type='weight', show_values=False)
axes[0, 0].set_title('1. Financial Fraud Drivers', fontsize=12)
axes[0, 0].set_xlabel('F-Score (Importance)')

# 2. Customer Analytics (Churn)
xgb.plot_importance(churn_model, ax=axes[0, 1], max_num_features=5, importance_type='weight', show_values=False)
axes[0, 1].set_title('2. Customer Churn Attrition Drivers', fontsize=12)
axes[0, 1].set_xlabel('F-Score (Importance)')

# 3. Operational Forecasting
xgb.plot_importance(forecast_model, ax=axes[1, 0], max_num_features=5, importance_type='weight', show_values=False)
axes[1, 0].set_title('3. Payout Forecasting Drivers', fontsize=12)
axes[1, 0].set_xlabel('F-Score (Importance)')

# 4. Healthcare & Diagnostics
xgb.plot_importance(health_model, ax=axes[1, 1], max_num_features=5, importance_type='weight', show_values=False)
axes[1, 1].set_title('4. Chronic Illness Risk Drivers', fontsize=12)
axes[1, 1].set_xlabel('F-Score (Importance)')

plt.tight_layout()
plt.subplots_adjust(top=0.90)

# Save the plot for local folder or your GitHub portfolio repository
plt.savefig('xgboost_feature_importance.png', dpi=300)
plt.show()


# -------------------------------------------------------------------------------
# Helper code to get raw percentage impact for your project write-up
def print_top_features(model, model_name):
    importance = model.get_booster().get_score(importance_type='weight')
    total = sum(importance.values())
    sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)

    print(f"\n--- {model_name} Top Drivers ---")
    for feat, score in sorted_features:
        pct = (score / total) * 100
        print(f"Feature '{feat}': Responsible for {pct:.1f}% of model decisions.")


print_top_features(fraud_model, "Fraud Detection")
print_top_features(churn_model, "Customer Churn")