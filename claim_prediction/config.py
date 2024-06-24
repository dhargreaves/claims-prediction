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
xgb_params = {'objective': 'binary:logistic',
                'base_score': None,
                'booster': None,
                'colsample_bylevel': None,
                'colsample_bynode': None,
                'colsample_bytree': np.float64(0.7257222527243534),
                'device': None,
                'eval_metric': ['auc', 'rmse', 'logloss'],
                'gamma': None,
                'grow_policy': None,
                'interaction_constraints': None,
                'learning_rate': np.float64(0.02592809939305637),
                'max_bin': None,
                'max_cat_threshold': None,
                'max_cat_to_onehot': None,
                'max_delta_step': None,
                'max_depth': 6,
                'max_leaves': None,
                'min_child_weight': 9,
                'monotone_constraints': None,
                'multi_strategy': None,
                'n_estimators':290,
                'n_jobs': None,
                'num_parallel_tree': None,
                'random_state': None,
                'reg_alpha': None,
                'reg_lambda': None,
                'sampling_method': None,
                'subsample': np.float64(0.6908946289879279),
                'tree_method': None,
                'validate_parameters': None,
                'verbosity': None}

target = 'claim_status'
inference_write_path =  ''