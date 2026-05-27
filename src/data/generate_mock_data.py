import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_poultry_data(num_batches=2000):
    """
    Generates realistic synthetic data for broiler chicken production.
    """
    np.random.seed(42) # For reproducibility
    
    # Generate IDs and Categories
    batch_ids = [f"LOTE_{str(i).zfill(5)}" for i in range(1, num_batches + 1)]
    farm_ids = np.random.choice([f"GRANJA_{str(i).zfill(2)}" for i in range(1, 15)], num_batches)
    farmer_ids = np.random.choice([f"GRANJEIRO_{str(i).zfill(2)}" for i in range(1, 20)], num_batches)
    supervisor_ids = np.random.choice(["SUP_A", "SUP_B", "SUP_C"], num_batches)
    
    strains = np.random.choice(["Cobb 500", "Ross 308"], num_batches)
    sexes = np.random.choice(["Macho", "Fêmea", "Misto"], num_batches)
    shed_types = np.random.choice(["Dark House", "Convencional"], num_batches, p=[0.7, 0.3])
    
    data = []
    
    for i in range(num_batches):
        # 1. Dates and Logistics
        placement_date = datetime(2024, 1, 1) + timedelta(days=np.random.randint(0, 365))
        slaughter_age = np.random.randint(42, 48) # Slaughter between 42 and 47 days
        slaughter_date = placement_date + timedelta(days=slaughter_age)
        
        flock_size = np.random.randint(20000, 35000)
        transport_distance = np.random.uniform(10.0, 150.0)
        fasting_time_hrs = np.random.choice([6, 8, 10, 12], p=[0.2, 0.6, 0.15, 0.05])
        mortality_rate = abs(np.random.normal(3.5, 1.5)) # Mean 3.5%
        
        # 2. Biology and Weights (Simulating the growth curve)
        # Factors that affect weight
        sex_multiplier = 1.05 if sexes[i] == "Macho" else (0.95 if sexes[i] == "Fêmea" else 1.0)
        shed_multiplier = 1.02 if shed_types[i] == "Dark House" else 0.98
        
        # Base weight at 30 days (~1.5kg to 1.7kg)
        base_30d = np.random.normal(1600, 80) * sex_multiplier * shed_multiplier
        
        # Daily weight gain after 30 days (~80g to 100g per day)
        daily_gain = np.random.normal(90, 5) * sex_multiplier * shed_multiplier
        
        weight_30d = base_30d
        weight_40d = weight_30d + (10 * daily_gain)
        
        # Actual final slaughter weight
        actual_slaughter_weight = weight_30d + ((slaughter_age - 30) * daily_gain)
        # Penalty for prolonged fasting and transport distance (weight loss)s
        actual_slaughter_weight -= (fasting_time_hrs * 5) + (transport_distance * 0.5) 
        
        # The Human Factor
        farmer_error = np.random.normal(0, 150) 
        farmer_estimated_weight = actual_slaughter_weight + farmer_error

        data.append({
            "batch_id": batch_ids[i],
            "farm_id": farm_ids[i],
            "farmer_id": farmer_ids[i],
            "supervisor_id": supervisor_ids[i],
            "placement_date": placement_date.strftime("%Y-%m-%d"),
            "slaughter_date": slaughter_date.strftime("%Y-%m-%d"),
            "strain": strains[i],
            "flock_sex": sexes[i],
            "shed_type": shed_types[i],
            "flock_size": flock_size,
            "transport_distance_km": round(transport_distance, 2),
            "fasting_time_hrs": fasting_time_hrs,
            "mortality_rate_pct": round(mortality_rate, 2),
            "weight_30d_g": round(weight_30d, 2),
            "weight_40d_g": round(weight_40d, 2),
            "farmer_estimated_weight_g": round(farmer_estimated_weight, 2),
            "actual_slaughter_weight_g": round(actual_slaughter_weight, 2)
        })

    df = pd.DataFrame(data)
    df.to_csv("historical_data.csv", index=False)
    print(f"Sucesso! Arquivo 'historical_data.csv' gerado com {num_batches} lotes simulados.")

if __name__ == "__main__":
    generate_poultry_data()
