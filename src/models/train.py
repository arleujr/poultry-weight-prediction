import duckdb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
import joblib
import os

def train_model():
    """
    Train an XGBoost regression model to predict actual slaughter weight
    using biological curves, feed consumption, and human estimations.
    """
    print("1. Connecting to Data Warehouse...")
    conn = duckdb.connect("data/processed/warehouse.duckdb")

    query = """
        SELECT 
            f.strain as lineage,
            f.flock_sex as sex,
            f.fasting_time_hrs,
            f.housed_birds,
            f.culls,
            f.mortality_rate_pct,
            (f.feed_delivered_kg - f.feed_remaining_kg) AS feed_consumed_kg,
            d.transport_distance_km as distance_km,
            date_diff('day', f.placement_date, f.slaughter_date) AS age_days,
            f.weight_7d_g as w_7,
            f.weight_14d_g as w_14,
            f.weight_21d_g as w_21,
            f.weight_28d_g as w_28,
            f.weight_35d_g as w_35,
            f.weight_42d_g as w_42,
            f.technician_estimated_weight_g as technician_estimate,
            f.actual_slaughter_weight_g
        FROM fct_slaughter_batches f
        JOIN dim_farm d ON f.farm_id = d.farm_id
    """
    df = conn.execute(query).df()
    conn.close()

    print("2. Preprocessing Data...")
    X = pd.get_dummies(df.drop(columns=['actual_slaughter_weight_g']), drop_first=True)
    y = df['actual_slaughter_weight_g']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("3. Training XGBoost Regressor...")
    model = xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    print("4. Evaluating Model Performance...")
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    print(f"   -> Mean Absolute Error (MAE): {mae:.2f} grams")

    if mae <= 100:
        print("   -> BUSINESS GOAL ACHIEVED: Error within +/- 100g margin!")
    else:
        print("   -> ATTENTION: Error above 100g margin. Model needs tuning.")

    print("\n5. Saving Model to disk...")
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/xgb_weight_predictor.joblib")
    joblib.dump(list(X.columns), "models/model_features.joblib") 
    
    print("Success! Model saved.")

if __name__ == "__main__":
    train_model()