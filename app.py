import streamlit as st
import pandas as pd
import numpy as np
import datetime
import joblib
import os

# -----------------------------------------------------------------------------
# 1. UI Configuration & Corporate Branding (Pif Paf Identity)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Preditor de Peso | Pif Paf",
    page_icon="🐔",
    layout="centered"
)

# Premium Custom CSS - Clean Dashboard & Rounded Cards Aesthetic
st.markdown("""
    <style>
    /* Fundo da aplicação levemente acinzentado para destacar os cards brancos */
    .stApp {
        background-color: #F8FAFC;
    }
    
    /* Linha de destaque superior com o Amarelo da marca */
    header {
        border-bottom: 5px solid #FFD200;
    }
    
    /* Título Principal no Vermelho oficial */
    h1 {
        color: #E30613 !important;
        font-weight: 700 !important;
    }
    
    /* Estilização dos blocos de métricas (Efeito Cards/Bento Grid) */
    div[data-testid="stMetricSimpleUnit"] {
        background-color: #FFFFFF;
        padding: 1.5rem;
        border-radius: 16px;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06);
        border-left: 5px solid #E30613;
    }
    
    /* Customização do Botão Principal */
    .stButton>button[kind="primary"] {
        background-color: #E30613 !important;
        color: #FFFFFF !important;
        border-radius: 12px !important;
        border: none !important;
        padding: 0.6rem 2rem !important;
        font-weight: 600 !important;
        width: 100%;
        transition: all 0.2s ease-in-out;
    }
    
    .stButton>button[kind="primary"]:hover {
        background-color: #FFD200 !important;
        color: #E30613 !important;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. Static Farm Registry (Master Data Representation)
# -----------------------------------------------------------------------------
FARM_REGISTRY = {
    "GRANJA_01": {"distance_km": 15.5},
    "GRANJA_02": {"distance_km": 45.5},
    "GRANJA_03": {"distance_km": 80.0}
}

# -----------------------------------------------------------------------------
# 3. Model Loading Components (Cached for performance optimization)
# -----------------------------------------------------------------------------
@st.cache_resource
def load_prediction_artifacts():
    """Loads serialized XGBoost model and corresponding feature structure."""
    model = joblib.load("models/xgb_weight_predictor.joblib")
    features = joblib.load("models/model_features.joblib")
    return model, features

try:
    model, expected_columns = load_prediction_artifacts()
except FileNotFoundError:
    st.error("Model artifacts not found. Please execute the training pipeline first.")
    st.stop()

# -----------------------------------------------------------------------------
# 4. Main Application Header
# -----------------------------------------------------------------------------
st.title("Previsão de peso de abate")
st.markdown("Insira os dados operacionais do lote para calcular a estimativa final baseada em aprendizado de máquina.")

# -----------------------------------------------------------------------------
# 5. Data Ingestion Form
# -----------------------------------------------------------------------------
with st.form("prediction_form"):
    st.subheader("Dados do lote atual")
    
    # Batch & Personnel Identification
    col_id1, col_id2, col_id3 = st.columns(3)
    
    with col_id2:
        farm_id = st.selectbox("ID da Granja", list(FARM_REGISTRY.keys()))
        
    with col_id1:
        hoje_str = datetime.datetime.now().strftime("%d%m")
        lote_sugerido = f"LOTE_{farm_id}_{hoje_str}"
        batch_id = st.text_input("ID do Lote", value=lote_sugerido)
        
    with col_id3:
        technician_id = st.selectbox("ID do Técnico", ["TECNICO_01", "TECNICO_02", "TECNICO_03"])
        
    # Logistics & Biological Parameters
    col_log1, col_log2 = st.columns(2)
    with col_log1:
        placement_date = st.date_input("Data de Alojamento", datetime.date(2026, 4, 1), format="DD/MM/YYYY")
        lineage = st.selectbox("Linhagem", ["Ross 308", "Cobb 500"])
        
        # Displaying the immutable distance fetched from master data
        st.info(f"Distância: {FARM_REGISTRY[farm_id]['distance_km']} km")
        
    with col_log2:
        slaughter_date = st.date_input("Data Prevista de Abate", datetime.date(2026, 5, 15), format="DD/MM/YYYY")
        sex = st.selectbox("Sexo do Lote", ["Macho", "Fêmea", "Misto"])
        fasting_hours = st.number_input("Tempo de Jejum (horas)", min_value=0.0, value=8.0, step=0.5)

    st.markdown("---")
    st.subheader("Manejo Sanitário e Consumo de Ração")
    
    # Health Metrics - UI collects raw data, backend calculates the rates
    col_mort1, col_mort2, col_mort3 = st.columns(3)
    with col_mort1:
        housed_birds = st.number_input("Total de Aves Alojadas", min_value=1, value=17200, step=100)
    with col_mort2:
        dead_birds = st.number_input("Aves Mortas no Núcleo", min_value=0, value=645, step=10)
    with col_mort3:
        culls = st.number_input("Refugos / Descartes", min_value=0, value=84, step=1)

    # Feed Efficiency Metrics
    col_feed1, col_feed2 = st.columns(2)
    with col_feed1:
        feed_delivered_kg = st.number_input("Ração Total Enviada (kg)", min_value=0, value=65000, step=500)
    with col_feed2:
        feed_remaining_kg = st.number_input("Sobra no Silo (kg)", min_value=0, value=1200, step=100)

    st.markdown("---")
    st.subheader("Curva Biológica de Crescimento (Pesagens)")
    
    # Time-series weight tracking
    col_w1, col_w2, col_w3 = st.columns(3)
    with col_w1:
        w_7 = st.number_input("Peso aos 7 dias (g)", value=201, step=5)
        w_28 = st.number_input("Peso aos 28 dias (g)", value=1671, step=10)
    with col_w2:
        w_14 = st.number_input("Peso aos 14 dias (g)", value=520, step=5)
        w_35 = st.number_input("Peso aos 35 dias (g)", value=2354, step=10)
    with col_w3:
        w_21 = st.number_input("Peso aos 21 dias (g)", value=1038, step=5)
        w_42 = st.number_input("Peso aos 42 dias (g)", value=3069, step=10)

    st.markdown("---")
    st.subheader("Estimativa final")
    
    # Human Baseline / Viable Target
    technician_estimate = st.number_input("Estimativa do Técnico (g)", min_value=0.0, value=3200.0, step=10.0)
    
    # Execution Trigger
    submit_button = st.form_submit_button("Calcular Estimativa", type="primary")

# -----------------------------------------------------------------------------
# 6. Inference Pipeline & Feature Alignment
# -----------------------------------------------------------------------------
if submit_button:
    # Feature Engineering: Feature derivations based on database schema definitions
    batch_age_days = (slaughter_date - placement_date).days
    
    # Cálculo corrigido: somando mortos + refugos antes de calcular a taxa
    mortality_rate = ((dead_birds + culls) / housed_birds) * 100 if housed_birds > 0 else 0
    
    feed_consumed_kg = feed_delivered_kg - feed_remaining_kg
    resolved_distance = FARM_REGISTRY[farm_id]['distance_km']
    
    # Creating raw feature dictionary matching the model training inputs
    input_data = {
        'lineage': [lineage],
        'sex': [sex],
        'fasting_time_hrs': [fasting_hours],
        'housed_birds': [housed_birds],
        'culls': [culls],
        'mortality_rate_pct': [mortality_rate],
        'feed_consumed_kg': [feed_consumed_kg],
        'distance_km': [resolved_distance],
        'age_days': [batch_age_days],
        'w_7': [w_7], 'w_14': [w_14], 'w_21': [w_21], 
        'w_28': [w_28], 'w_35': [w_35], 'w_42': [w_42],
        'technician_estimate': [technician_estimate]
    }
    
    input_df = pd.DataFrame(input_data)
    
    # Handling categorical cross-sections via One-Hot Encoding alignment
    input_encoded = pd.get_dummies(input_df)
    input_aligned = input_encoded.reindex(columns=expected_columns, fill_value=0)
    
    # Model Inference execution
    prediction = model.predict(input_aligned)[0]
    adjustment = prediction - technician_estimate
    
    # -------------------------------------------------------------------------
    # 7. Output Visualization Layer (Estilo Cards Premium)
    # -------------------------------------------------------------------------
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Análise de Série Temporal com Contexto Local concluída com sucesso.")
    
    col_res1, col_res2, col_res3 = st.columns(3)
    
    with col_res1:
        st.metric(
            label="Estimativa de Campo", 
            value=f"{technician_estimate:.0f} g",
            help="Peso médio estimado visualmente pelo técnico"
        )
        
    with col_res2:
        st.metric(
            label="Previsão Inteligente (IA)", 
            value=f"{prediction:.0f} g", 
            delta=f"{adjustment:.0f} g de ajuste", 
            delta_color="inverse"
        )
        
    with col_res3:
        st.metric(
            label="Mortalidade Real", 
            value=f"{mortality_rate:.2f}%",
            help="Índice combinando aves mortas e refugos do lote."
        )
        
    st.markdown("<br>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # 8. Data Persistence for BI Synchronization
    # -------------------------------------------------------------------------
    output_dir = "data/processed"
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "ai_predictions.csv")
    
    # Appending session metadata for audit logs
    input_df['batch_id'] = batch_id
    input_df['farm_id'] = farm_id
    input_df['technician_id'] = technician_id
    input_df['ai_predicted_weight'] = prediction
    input_df['timestamp'] = datetime.datetime.now()
    
    if os.path.exists(csv_path):
        input_df.to_csv(csv_path, mode='a', header=False, index=False)
    else:
        input_df.to_csv(csv_path, index=False)
        
    st.info("Log de predição gravado com sucesso.")

# -----------------------------------------------------------------------------
# 9. Developer Signature / Footer Module
# -----------------------------------------------------------------------------
st.markdown("<br><br><br>", unsafe_allow_html=True)
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        <small>
            Desenvolvido por <b>Arleu Junior</b> 
        </small>
    </div>
    """, 
    unsafe_allow_html=True
)