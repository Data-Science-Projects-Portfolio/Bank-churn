# Churn Prediction Project: Final Analysis & Performance Notes

## 1. Data Exploration & Baseline Assessment
- The dataset exhibits a significant class imbalance with a ~20.4% churn rate (2,037 churned customers out of 10,000 total records).
- An initial baseline Logistic Regression model reached an overall accuracy of 81%, but struggled significantly on the target positive class, showing a poor **Recall of only 19%** and a **ROC-AUC of 0.7745**.

## 2. Advanced Classifier Benchmarking
Four distinct modeling pipelines were trained and evaluated on the held-out test set under default settings:
- **Logistic Regression (Class-balanced):** Recall 70.52%, ROC-AUC 0.7769
- **Random Forest (Class-balanced):** Recall 42.26%, ROC-AUC 0.8483
- **XGBoost (Weighted):** Recall 61.18%, ROC-AUC 0.8282
- **Gradient Boosting:** Recall 50.12%, ROC-AUC 0.8706

While the **Gradient Boosting** classifier possessed the highest overall discriminative power with a **ROC-AUC of 0.8706**, its default classification threshold of 0.5 yielded an insufficient recall of 50.12%, missing half of the customers likely to churn.

## 3. Threshold Calibration for Churn Catchment
To prioritize high catchment of at-risk customers, a decision threshold optimization was executed on a separate validation split:
- By establishing a target validation recall constraint of **80%**, the optimal probability threshold was identified at **0.1583**.
- Evaluating the final model on the independent test set with this customized threshold successfully shifted the classification metrics:
  - **Test Recall (Churn): 83.0%** (exceeding the 80% business target)
  - **Test ROC-AUC: 0.8706**
  - **F1-Score (Churn): 0.58**

## 4. Model Interpretability & Business Drivers
A SHAP TreeExplainer analyzed feature attributions on the test predictions to capture key churn motivators:
- **Number of Products (products_number):** This is the strongest driver. Customers holding 3 or 4 products show exceptionally high propensity to churn.
- **Age (age):** Higher age groups are consistently linked with higher churn likelihood.
- **Active Membership (active_member):** Inactive accounts correlate with a notable increase in churn predictions.
- **Geographic Location (country_Germany):** Being located in Germany is associated with a distinct positive impact on churn risk compared to Spain and France.