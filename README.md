# 📈 Health Insurance Optimization Suite

## Project Overview
Built a multi-task analytics pipeline using **Python** and **XGBoost** to simultaneously mitigate operational risk, minimize client attrition, project financial liabilities, and proactively identify high-risk patient segments.
- **Fraud Mitigation:** Classified anomalous claim behaviors, saving a projected **$X** in simulated leakages.
- **Retention Engine:** Flagged churn-vulnerable accounts with a macro F1-score of ***X%***.
- **Capital Forecasting:** Reduced reserves forecasting error to an RMSE of just **±$*X***.

### 📊 Model Interpretability & Business Drivers

To ensure transparency for non-technical stakeholders (e.g., risk officers and medical directors), I extracted feature importance metrics using the XGBoost F-Score (weight-based split frequency).

![XGBoost Feature Importance](images/xgboost_feature_importance.png)

#### **Key Business Insights Derived:**
1. **Fraud Mitigation:** Claim amount history and distance from home emerged as the primary signals, allowing the risk team to automatically flag anomalous high-dollar out-of-network claims.
2. **Customer Retention:** Customer service call volume heavily out-weighted premium price as a churn indicator, shifting corporate strategy from price-discounting to customer-support optimization.
3. **Operational Forecasting:** Age and pre-existing conditions dictate 80%+ of next-month financial liabilities, enabling precise liquid capital budgeting.
4. **Healthcare Stratification:** BMI and aging thresholds reliably map out preventive health needs, allowing early intervention outreach.
