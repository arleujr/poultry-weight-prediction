import duckdb

def build_star_schema():
    """
    Transform raw staging data into a star schema (fact and dimension tables)
    optimized for analytics and machine learning.
    """
    db_path = "data/processed/warehouse.duckdb"
    print("Connecting to data warehouse...")
    conn = duckdb.connect(db_path)

    try:
        # Dimension: farms (stable farm attributes for joins and filters)
        print("Building dim_farm...")
        conn.execute("""
            CREATE OR REPLACE TABLE dim_farm AS
            SELECT DISTINCT
                farm_id,
                shed_type,
                transport_distance_km
            FROM raw_batches;
        """)

        # Dimension: farmers (canonical list for joins and enrichment)
        print("Building dim_farmer...")
        conn.execute("""
            CREATE OR REPLACE TABLE dim_farmer AS
            SELECT DISTINCT farmer_id
            FROM raw_batches;
        """)

        # Dimension: supervisors (canonical list for accountability)
        print("Building dim_supervisor...")
        conn.execute("""
            CREATE OR REPLACE TABLE dim_supervisor AS
            SELECT DISTINCT supervisor_id
            FROM raw_batches;
        """)

        # Fact: slaughter batches (event-level metrics and foreign keys)
        print("Building fct_slaughter_batches...")
        conn.execute("""
            CREATE OR REPLACE TABLE fct_slaughter_batches AS
            SELECT
                batch_id,
                farm_id,
                farmer_id,
                supervisor_id,
                CAST(placement_date AS DATE) AS placement_date,
                CAST(slaughter_date AS DATE) AS slaughter_date,
                strain,
                flock_sex,
                flock_size,
                mortality_rate_pct,
                fasting_time_hrs,
                weight_30d_g,
                weight_40d_g,
                farmer_estimated_weight_g,
                actual_slaughter_weight_g,
                (farmer_estimated_weight_g - actual_slaughter_weight_g) AS farmer_error_margin_g
            FROM raw_batches;
        """)

        print("Star schema built successfully.")

        # Validation: list created tables
        tables = conn.execute("SHOW TABLES;").df()
        print("\nCurrent tables in the data warehouse:")
        print(tables)

    except Exception as e:
        print(f"Error during transformation: {e}")
    finally:
        conn.close()
        print("Database connection closed.")

if __name__ == "__main__":
    build_star_schema()
