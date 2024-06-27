from ..data_processing.load_data import TrainingDataLoader

def test_training_data_loader():
    training_data_loader = TrainingDataLoader()
    training_data = training_data_loader.load_data()
    expected_dtypes = {
        'claim_status': 'int64',
        'age': 'int64',
        'height_cm': 'int64',
        'weight_kg': 'int64',
        'income': 'int64',
        'financial_hist_1': 'float64',
        'financial_hist_2': 'float64',
        'financial_hist_3': 'float64',
        'financial_hist_4': 'float64',
        'credit_score_1': 'int64',
        'credit_score_2': 'int64',
        'credit_score_3': 'int64',
        'insurance_hist_1': 'float64',
        'insurance_hist_2': 'float64',
        'insurance_hist_3': 'float64',
        'insurance_hist_4': 'float64',
        'insurance_hist_5': 'float64',
        'bmi': 'int64',
        'gender': 'int64',
        'marital_status': 'object',
        'occupation': 'object',
        'location': 'object',
        'prev_claim_rejected': 'int64',
        'known_health_conditions': 'int64',
        'uk_residence': 'int64',
        'family_history_1': 'int64',
        'family_history_2': 'int64',
        'family_history_3': 'object',
        'family_history_4': 'int64',
        'family_history_5': 'int64',
        'product_var_1': 'int64',
        'product_var_2': 'int64',
        'product_var_3': 'object',
        'product_var_4': 'int64',
        'health_status': 'int64',
        'driving_record': 'int64',
        'previous_claim_rate': 'int64',
        'education_level': 'int64',
        'income level': 'int64',
        'n_dependents': 'int64',
        'employment_type': 'object',
    }

    assert set(training_data.columns) == set(expected_dtypes.keys())
    for column, dtype in expected_dtypes.items():
        assert training_data[column].dtype == dtype

