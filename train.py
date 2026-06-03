import pandas as pd
import numpy as np
import joblib
import os
from xgboost import XGBRegressor

# -----------------------------------------------------------------------------
# 1. Configuração de Sementes e Amostragem (Dados Simulados)
# -----------------------------------------------------------------------------
np.random.seed(42)
n_samples = 1000

# Gerando a base simétrica à regra de negócio do web app
data = {
    'lineage': np.random.choice(['Ross 308', 'Cobb 500'], n_samples),
    'sex': np.random.choice(['Macho', 'Fêmea', 'Misto'], n_samples),
    'fasting_time_hrs': np.random.uniform(4, 12, n_samples),
    'housed_birds': np.random.randint(15000, 35000, n_samples),
    'culls': np.random.randint(10, 150, n_samples),
    'mortality_rate_pct': np.random.uniform(1.5, 6.0, n_samples),
    'feed_consumed_kg': np.random.uniform(50000, 120000, n_samples),
    'distance_km': np.random.choice([15.5, 45.5, 80.0], n_samples),
    'age_days': np.random.randint(40, 48, n_samples),
    
    # Histórico de Série Temporal de Pesagens (w_7 até w_42)
    'w_7': np.random.normal(200, 15, n_samples),
    'w_14': np.random.normal(515, 30, n_samples),
    'w_21': np.random.normal(1020, 50, n_samples),
    'w_28': np.random.normal(1650, 80, n_samples),
    'w_35': np.random.normal(2320, 100, n_samples),
    'w_42': np.random.normal(3050, 120, n_samples),
    
    # Estimativa Humana Base
    'technician_estimate': np.random.normal(3150, 130, n_samples)
}

df = pd.DataFrame(data)

# -----------------------------------------------------------------------------
# 2. Regra Biológica Alvo (Simulando o Peso Real de Abate no Frigorífico)
# -----------------------------------------------------------------------------
# O peso real correlaciona fortemente com o w_42, consumo de ração e penalidade de jejum/distância
df['actual_weight'] = (
    df['w_42'] * 0.85 + 
    (df['feed_consumed_kg'] / df['housed_birds']) * 280 - 
    (df['fasting_time_hrs'] * 8) - 
    (df['distance_km'] * 0.15) + 
    np.random.normal(0, 30, n_samples)
)

# Separando preditores (X) e alvo (y)
X = df.drop(columns=['actual_weight'])
y = df['actual_weight']

# -----------------------------------------------------------------------------
# 3. Alinhamento de Dummies (Idêntico ao processo do app.py)
# -----------------------------------------------------------------------------
X_encoded = pd.get_dummies(X)
model_features = X_encoded.columns.tolist()

# -----------------------------------------------------------------------------
# 4. Treinamento Estruturado do XGBoost
# -----------------------------------------------------------------------------
model = XGBRegressor(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42
)

model.fit(X_encoded, y)
print("Treinamento concluído com sucesso!")

# -----------------------------------------------------------------------------
# 5. Exportação Segura dos Artefatos para Produção
# -----------------------------------------------------------------------------
os.makedirs('models', exist_ok=True)

# Salvando exatamente o que o app.py tenta ler
joblib.dump(model, 'models/xgb_weight_predictor.joblib')
joblib.dump(model_features, 'models/model_features.joblib')

print("Artefatos salvos na pasta 'models/' com sucesso!")