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
    using historical flock data and farmer estimations.
    """
    print("1. Connecting to Data Warehouse...")
    conn = duckdb.connect("data/processed/warehouse.duckdb")

    # Query: join fact table with farm dimension
    query = """
        SELECT 
            f.weight_30d_g,
            f.weight_40d_g,
            f.farmer_estimated_weight_g,
            f.flock_sex,
            f.fasting_time_hrs,
            f.mortality_rate_pct,
            d.transport_distance_km,
            d.shed_type,
            date_diff('day', f.placement_date, f.slaughter_date) AS slaughter_age_days,
            f.actual_slaughter_weight_g
        FROM fct_slaughter_batches f
        JOIN dim_farm d ON f.farm_id = d.farm_id
    """
    df = conn.execute(query).df()
    conn.close()

    print("2. Preprocessing Data...")
    # Convert categorical variables into dummy/indicator variables
    X = pd.get_dummies(df.drop(columns=['actual_slaughter_weight_g']), drop_first=True)
    y = df['actual_slaughter_weight_g']

    # Split dataset: 80% train, 20% test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("3. Training XGBoost Regressor...")
    # Configure and fit model
    model = xgb.XGBRegressor(
        n_estimators=200,     # Number of boosting rounds
        learning_rate=0.05,   # Step size shrinkage
        max_depth=5,          # Maximum tree depth
        random_state=42
    )
    model.fit(X_train, y_train)

    print("4. Evaluating Model Performance...")
    predictions = model.predict(X_test)
    
    # MAE: average absolute error in grams
    mae = mean_absolute_error(y_test, predictions)
    # RMSE: penalizes large errors more heavily
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    
    print(f"   -> Mean Absolute Error (MAE): {mae:.2f} grams")
    print(f"   -> Root Mean Squared Error (RMSE): {rmse:.2f} grams")

    # Business KPI check
    if mae <= 100:
        print("   -> BUSINESS GOAL ACHIEVED: Error within +/- 100g margin!")
    else:
        print("   -> ATTENTION: Error above 100g margin. Model needs tuning.")

    print("\n5. Saving Model to disk...")
    os.makedirs("models", exist_ok=True)
    
    # Save trained model
    joblib.dump(model, "models/xgb_weight_predictor.joblib")
    # Save feature order (needed for inference consistency)
    joblib.dump(list(X.columns), "models/model_features.joblib") 
    
    print("Success! Model saved at 'models/xgb_weight_predictor.joblib'")

if __name__ == "__main__":
    train_model()
