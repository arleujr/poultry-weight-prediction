import pandas as pd
import joblib
import os

def run_inference():
    print("1. Reading current batch data...")
    # Load supervisor input file
    input_data = pd.read_csv("data/input/supervisor_input.csv")
    
    # Calculate slaughter age in days
    input_data['placement_date'] = pd.to_datetime(input_data['placement_date'])
    input_data['slaughter_date'] = pd.to_datetime(input_data['slaughter_date'])
    input_data['slaughter_age_days'] = (input_data['slaughter_date'] - input_data['placement_date']).dt.days

    # Select features used by the model
    features = input_data[[
        'weight_30d_g', 'weight_40d_g', 'farmer_estimated_weight_g', 
        'flock_sex', 'fasting_time_hrs', 'mortality_rate_pct',
        'transport_distance_km', 'shed_type', 'slaughter_age_days'
    ]]

    print("2. Encoding categorical variables...")
    # Convert categorical values into dummy/indicator variables
    features_encoded = pd.get_dummies(features)

    print("3. Loading trained model...")
    model = joblib.load("models/xgb_weight_predictor.joblib")
    expected_columns = joblib.load("models/model_features.joblib")

    # Align columns with training set (fill missing with 0)
    features_aligned = features_encoded.reindex(columns=expected_columns, fill_value=0)

    print("4. Predicting slaughter weight...")
    prediction = model.predict(features_aligned)
    
    input_data['AI_Predicted_Weight_g'] = prediction
    
    print("\n" + "="*50)
    print(" INFERENCE RESULT")
    print("="*50)
    print(f"Batch: {input_data['batch_id'].values[0]}")
    print(f"Slaughter Age: {input_data['slaughter_age_days'].values[0]} days")
    print(f"Farmer Estimate: {input_data['farmer_estimated_weight_g'].values[0]}g")
    print(f"AI Prediction: {prediction[0]:.2f}g")
    
    # Ensure output directory exists
    os.makedirs("data/processed", exist_ok=True)
    # Save results for dashboard integration
    input_data.to_csv("data/processed/ai_predictions.csv", index=False)
    print("="*50)
    print("Result saved to dashboard file (data/processed/ai_predictions.csv)")

if __name__ == "__main__":
    run_inference()
