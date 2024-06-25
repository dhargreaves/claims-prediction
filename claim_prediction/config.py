import numpy as np

# pre-processing
non_numerical_columns = ['gender', 'marital_status', 'occupation', 'location', 'prev_claim_rejected', 'known_health_conditions', 'uk_residence', 'family_history_1', 'family_history_2', 'family_history_4', 'family_history_5', 'product_var_1', 'product_var_2', 'product_var_3', 'health_status', 'driving_record', 'previous_claim_rate', 'education_level', 'income level', 'n_dependents']

features = ['age', 'height_cm', 'weight_kg', 'income', 'financial_hist_1',
       'financial_hist_2', 'financial_hist_3', 'financial_hist_4',
       'credit_score_1', 'credit_score_2', 'credit_score_3',
       'insurance_hist_1', 'insurance_hist_2', 'insurance_hist_3',
       'insurance_hist_4', 'insurance_hist_5', 'bmi', 'gender',
       'marital_status', 'occupation', 'location', 'prev_claim_rejected',
       'known_health_conditions', 'uk_residence', 'family_history_1',
       'family_history_2', 'family_history_4', 'family_history_5',
       'product_var_1', 'product_var_2', 'product_var_3', 'product_var_4',
       'health_status', 'driving_record', 'previous_claim_rate',
       'education_level', 'income level', 'n_dependents']

# models
eval_metrics = ['auc', 'rmse', 'logloss']
target = 'claim_status'
inference_write_path =  ''