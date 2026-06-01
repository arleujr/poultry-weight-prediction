import duckdb
import pandas as pd
import numpy as np
import os
import datetime

def generate_mock_raw_data():
    """Gera o arquivo raw_batches.csv com os novos dados biológicos e de ração."""
    print("Generating synthetic raw data with biological curves...")
    np.random.seed(42)
    num_records = 500
    
    farms = ["GRANJA_01", "GRANJA_02", "GRANJA_03"]
    farm_distances = {"GRANJA_01": 15.5, "GRANJA_02": 45.5, "GRANJA_03": 80.0}
    
    data = {
        'batch_id': [f"LOTE_{i:04d}" for i in range(num_records)],
        'farm_id': np.random.choice(farms, num_records),
        'technician_id': np.random.choice(["TECNICO_01", "TECNICO_02", "TECNICO_03"], num_records),
        'placement_date': [datetime.date(2025, 1, 1) + datetime.timedelta(days=int(x)) for x in np.random.randint(0, 300, num_records)],
        'strain': np.random.choice(["Ross 308", "Cobb 500"], num_records),
        'flock_sex': np.random.choice(["Macho", "Fêmea", "Misto"], num_records),
        'fasting_time_hrs': np.random.uniform(6.0, 12.0, num_records),
        'housed_birds': np.random.randint(15000, 30000, num_records),
        'culls': np.random.randint(30, 150, num_records)
    }
    
    df = pd.DataFrame(data)
    df['transport_distance_km'] = df['farm_id'].map(farm_distances)
    df['shed_type'] = np.where(df['farm_id'] == "GRANJA_01", "Convencional", "Dark House")
    df['slaughter_date'] = df['placement_date'] + pd.to_timedelta(np.random.randint(42, 48, num_records), unit='d')
    
    df['mortality_rate_pct'] = np.random.uniform(1.5, 5.5, num_records)
    
    # Pesagens
    df['weight_7d_g'] = np.random.normal(200, 10, num_records)
    df['weight_14d_g'] = df['weight_7d_g'] + np.random.normal(320, 15, num_records)
    df['weight_21d_g'] = df['weight_14d_g'] + np.random.normal(510, 20, num_records)
    df['weight_28d_g'] = df['weight_21d_g'] + np.random.normal(630, 25, num_records)
    df['weight_35d_g'] = df['weight_28d_g'] + np.random.normal(700, 30, num_records)
    df['weight_42d_g'] = df['weight_35d_g'] + np.random.normal(720, 35, num_records)
    
    # Peso real (com perdas) e Ração
    df['actual_slaughter_weight_g'] = df['weight_42d_g'] + 150 - (df['fasting_time_hrs'] * 2.5) - (df['transport_distance_km'] * 0.1)
    df['feed_delivered_kg'] = ((df['housed_birds'] * df['actual_slaughter_weight_g']) / 1000) * np.random.uniform(1.5, 1.7, num_records) + 500
    df['feed_remaining_kg'] = np.random.uniform(100, 1000, num_records)
    
    # Viés do técnico
    df['technician_estimated_weight_g'] = df['actual_slaughter_weight_g'] + np.random.normal(50, 60, num_records)
    
    os.makedirs("data/raw", exist_ok=True)
    df.to_csv("data/raw/raw_batches.csv", index=False)

def build_star_schema():
    """
    Transform raw staging data into a star schema (fact and dimension tables)
    optimized for analytics and machine learning.
    """
    generate_mock_raw_data()
    
    db_path = "data/processed/warehouse.duckdb"
    os.makedirs("data/processed", exist_ok=True)
    print("\nConnecting to data warehouse...")
    conn = duckdb.connect(db_path)

    try:
        print("Building dim_farm...")
        conn.execute("""
            CREATE OR REPLACE TABLE dim_farm AS
            SELECT DISTINCT farm_id, shed_type, transport_distance_km
            FROM read_csv_auto('data/raw/raw_batches.csv');
        """)

        print("Building dim_technician...")
        conn.execute("""
            CREATE OR REPLACE TABLE dim_technician AS
            SELECT DISTINCT technician_id
            FROM read_csv_auto('data/raw/raw_batches.csv');
        """)

        print("Building fct_slaughter_batches...")
        conn.execute("""
            CREATE OR REPLACE TABLE fct_slaughter_batches AS
            SELECT
                batch_id, farm_id, technician_id,
                CAST(placement_date AS DATE) AS placement_date,
                CAST(slaughter_date AS DATE) AS slaughter_date,
                strain, flock_sex, housed_birds, culls, mortality_rate_pct,
                feed_delivered_kg, feed_remaining_kg, fasting_time_hrs,
                weight_7d_g, weight_14d_g, weight_21d_g, weight_28d_g, weight_35d_g, weight_42d_g,
                technician_estimated_weight_g, actual_slaughter_weight_g,
                (technician_estimated_weight_g - actual_slaughter_weight_g) AS error_margin_g
            FROM read_csv_auto('data/raw/raw_batches.csv');
        """)

        print("Star schema built successfully.")
    except Exception as e:
        print(f"Error during transformation: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    build_star_schema()