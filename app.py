import streamlit as st
import pandas as pd
import numpy as np
import datetime
import joblib
import os

# -----------------------------------------------------------------------------
# 1. UI Configuration & Advanced Mobile Responsive Aesthetic
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Preditor de Peso",
    page_icon="🐔",
    layout="centered"
)

# Custom CSS - Ajuste responsivo perfeito sem quebrar as bordas laterais
st.markdown("""
    <style>
    /* Fundo geral moderno */
    .stApp {
        background-color: #F8FAFC;
    }
    
    /* Header Dark Mode Premium Compactado */
    .brand-banner {
        background: linear-gradient(135deg, #0B1528 0%, #1E293B 100%);
        padding: 1.5rem;
        border-radius: 16px;
        margin-bottom: 1rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
        border-bottom: 4px solid #FFD200;
    }
    
    .brand-banner h1 {
        color: #FFFFFF !important;
        font-size: 1.6rem !important;
        font-weight: 700 !important;
        margin-bottom: 0.2rem !important;
    }
    
    .brand-banner p {
        color: #94A3B8 !important;
        font-size: 0.85rem !important;
        margin: 0 !important;
    }
    
    /* Estilização das Abas (Tabs) Compactas */
    .stTabs [data-baseweb="tab-list"] {
        gap: 5px;
        background-color: #E2E8F0;
        padding: 4px;
        border-radius: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 35px;
        padding-left: 8px !important;
        padding-right: 8px !important;
        font-size: 0.85rem !important;
        background-color: transparent;
        border-radius: 6px;
        color: #64748B;
        font-weight: 600;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        color: #E30613 !important;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #FFFFFF !important;
        color: #E30613 !important;
    }
    
    .stTabs [data-baseweb="tab-highlight-bar"] {
        background-color: #E30613 !important;
    }

    /* Customização dos inputs menos espessos */
    div[data-baseweb="select"], div[data-baseweb="input"], input {
        border-radius: 10px !important;
        border: 1px solid #E2E8F0 !important;
        background-color: #FFFFFF !important;
        height: 42px !important;
    }
    
    div[data-baseweb="select"]:focus-within, div[data-baseweb="input"]:focus-within {
        border-color: #E30613 !important;
        box-shadow: 0 0 0 3px rgba(227, 6, 19, 0.1) !important;
    }
    
    /* Ajuste de Margens de Subtítulos */
    .stSubheader h3 {
        font-size: 1.1rem !important;
        margin-bottom: 0.5rem !important;
    }
    
    /* ------------------------------------------------------------------------
       FLEX BOX ADAPTÁVEL: PREVINE ROLAGEM LATERAL EM QUALQUER CELULAR
       ------------------------------------------------------------------------ */
    @media (max-width: 640px) {
        div[data-testid="stHorizontalBlock"] {
            display: flex !important;
            flex-direction: row !important;
            flex-wrap: wrap !important; /* Permite quebra se faltar espaço físico */
            gap: 8px !important;
        }
        div[data-testid="column"] {
            flex: 1 1 45% !important; /* Ocupa metade da tela se necessário, sem estourar */
            min-width: 130px !important; /* Não deixa esmagar o texto */
        }
        /* Ajuste fino das métricas internas para telas mini */
        div[data-testid="stMetricSimpleUnit"] {
            padding: 0.7rem !important;
            border-radius: 12px !important;
            border-left: 4px solid #E30613 !important;
        }
        div[data-testid="stMetricLabel"] {
            font-size: 0.75rem !important;
        }
        div[data-testid="stMetricValue"] {
            font-size: 1.05rem !important;
        }
        div[data-testid="stMetricDelta"] {
            font-size: 0.65rem !important;
        }
        .kpi-box {
            padding: 0.7rem !important;
            border-radius: 12px !important;
        }
        .kpi-box h3 {
            font-size: 0.95rem !important;
        }
    }

    /* Desktop View Padronizada */
    div[data-testid="stMetricSimpleUnit"] {
        background-color: #FFFFFF;
        padding: 1.5rem;
        border-radius: 16px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        border-left: 6px solid #E30613;
    }
    
    /* Componente de Informação st.info */
    div[data-testid="stNotification"] {
        background-color: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-left: 5px solid #E30613 !important;
        border-radius: 12px !important;
        padding: 0.75rem !important;
    }
    div[data-testid="stNotification"] p {
        color: #E30613 !important;
        font-weight: 700 !important;
        font-size: 0.9rem !important;
    }
    div[data-testid="stNotification"] svg {
        fill: #E30613 !important;
        color: #E30613 !important;
    }
    
    /* Cards de KPI de Suporte Lateral */
    .kpi-box {
        background-color: #FFFFFF;
        padding: 1.2rem;
        border-radius: 16px;
        border: 1px solid #E2E8F0;
        border-left: 4px solid #FFD200;
        margin-bottom: 0.5rem;
    }
    
    /* Botão Principal */
    .stButton>button[kind="primary"] {
        background-color: #E30613 !important;
        color: #FFFFFF !important;
        border-radius: 12px !important;
        border: none !important;
        padding: 0.8rem 2rem !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        width: 100%;
        box-shadow: 0 4px 10px rgba(227, 6, 19, 0.2);
        transition: all 0.2s ease-in-out;
    }
    .stButton>button[kind="primary"]:hover {
        background-color: #B90510 !important;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. Static Farm Registry
# -----------------------------------------------------------------------------
FARM_REGISTRY = {
    "GRANJA_01": {"distance_km": 15.5},
    "GRANJA_02": {"distance_km": 45.5},
    "GRANJA_03": {"distance_km": 80.0}
}

# -----------------------------------------------------------------------------
# 3. Model Loading Components
# -----------------------------------------------------------------------------
@st.cache_resource
def load_prediction_artifacts():
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
st.markdown("""
    <div class="brand-banner">
        <h1>Previsão de peso de abate</h1>
        <p>Insira os dados operacionais do lote para calcular a estimativa final baseada em aprendizado de máquina.</p>
    </div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 5. Interactive Data Entry Flow (Texto Simplificado conforme solicitado)
# -----------------------------------------------------------------------------
st.subheader("Dados do Lote")

tab_log, tab_san, tab_weight = st.tabs(["📍 1. Lote", "🏥 2. Manejo", "📈 3. Pesagens"])

with tab_log:
    st.markdown("<br>", unsafe_allow_html=True)
    col_id1, col_id2, col_id3 = st.columns(3)
    with col_id1:
        farm_id = st.selectbox("ID da Granja", list(FARM_REGISTRY.keys()))
    with col_id2:
        hoje_str = datetime.datetime.now().strftime("%d%m")
        batch_id = st.text_input("ID do Lote", value=f"LOTE_{farm_id}_{hoje_str}")
    with col_id3:
        technician_id = st.selectbox("ID do Técnico", ["TECNICO_01", "TECNICO_02", "TECNICO_03"])
        
    col_dates1, col_dates2 = st.columns(2)
    with col_dates1:
        placement_date = st.date_input("Data de Alojamento", datetime.date(2026, 4, 1), format="DD/MM/YYYY")
        lineage = st.selectbox("Linhagem", ["Ross 308", "Cobb 500"])
        st.info(f"Distância: {FARM_REGISTRY[farm_id]['distance_km']} km")
    with col_dates2:
        slaughter_date = st.date_input("Data Prevista de Abate", datetime.date(2026, 5, 15), format="DD/MM/YYYY")
        sex = st.selectbox("Sexo do Lote", ["Macho", "Fêmea", "Misto"])
        fasting_hours = st.number_input("Tempo de Jejum (horas)", min_value=0.0, value=8.0, step=0.5)

with tab_san:
    st.markdown("<br>", unsafe_allow_html=True)
    col_mort1, col_mort2, col_mort3 = st.columns(3)
    with col_mort1:
        housed_birds = st.number_input("Total de Aves Alojadas", min_value=1, value=17200, step=100)
    with col_mort2:
        dead_birds = st.number_input("Aves Mortas no Núcleo", min_value=0, value=645, step=10)
    with col_mort3:
        culls = st.number_input("Refugos / Descartes", min_value=0, value=84, step=1)

    col_feed1, col_feed2 = st.columns(2)
    with col_feed1:
        feed_delivered_kg = st.number_input("Ração Total Enviada (kg)", min_value=0, value=65000, step=500)
    with col_feed2:
        feed_remaining_kg = st.number_input("Sobra no Silo (kg)", min_value=0, value=1200, step=100)

with tab_weight:
    st.markdown("<br>", unsafe_allow_html=True)
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
    technician_estimate = st.number_input("Estimativa do Técnico (g)", min_value=0.0, value=3200.0, step=10.0)

# -----------------------------------------------------------------------------
# 6. Real-Time Pre-Calculated KPI Indicators
# -----------------------------------------------------------------------------
st.markdown("<br>", unsafe_allow_html=True)
col_kpi1, col_kpi2 = st.columns(2)
batch_age_calc = (slaughter_date - placement_date).days
feed_consumed_calc = feed_delivered_kg - feed_remaining_kg

with col_kpi1:
    st.markdown(f"""
        <div class="kpi-box">
            <small style="color: #64748B; font-weight:600;">Idade do Ciclo</small>
            <h3 style="margin:0; color:#1E293B;">{batch_age_calc} dias</h3>
        </div>
    """, unsafe_allow_html=True)

with col_kpi2:
    st.markdown(f"""
        <div class="kpi-box">
            <small style="color: #64748B; font-weight:600;">Ração Consumida</small>
            <h3 style="margin:0; color:#1E293B;">{feed_consumed_calc:,} kg</h3>
        </div>
    """, unsafe_allow_html=True)

# Main Execution Trigger
submit_button = st.button("Executar Previsão", type="primary")

# -----------------------------------------------------------------------------
# 7. Inference Pipeline & Output Layout
# -----------------------------------------------------------------------------
if submit_button:
    mortality_rate = ((dead_birds + culls) / housed_birds) * 100 if housed_birds > 0 else 0
    resolved_distance = FARM_REGISTRY[farm_id]['distance_km']
    
    input_data = {
        'lineage': [lineage], 'sex': [sex], 'fasting_time_hrs': [fasting_hours],
        'housed_birds': [housed_birds], 'culls': [culls], 'mortality_rate_pct': [mortality_rate],
        'feed_consumed_kg': [feed_consumed_calc], 'distance_km': [resolved_distance],
        'age_days': [batch_age_calc], 'w_7': [w_7], 'w_14': [w_14], 'w_21': [w_21], 
        'w_28': [w_28], 'w_35': [w_35], 'w_42': [w_42], 'technician_estimate': [technician_estimate]
    }
    
    input_df = pd.DataFrame(input_data)
    input_encoded = pd.get_dummies(input_df)
    input_aligned = input_encoded.reindex(columns=expected_columns, fill_value=0)
    
    prediction = model.predict(input_aligned)[0]
    adjustment = prediction - technician_estimate
    
    st.markdown("---")
    st.subheader("Análise concluída.")
    
    col_res1, col_res2, col_res3 = st.columns(3)
    with col_res1:
        st.metric(label="Estimativa Campo", value=f"{technician_estimate:.0f} g")
    with col_res2:
        st.metric(label="Previsão IA", value=f"{prediction:.0f} g", delta=f"{adjustment:.0f} g", delta_color="inverse")
    with col_res3:
        st.metric(label="Mortalidade Real", value=f"{mortality_rate:.2f}%")
        
    st.markdown("<br>", unsafe_allow_html=True)

    # Persistence Module
    output_dir = "data/processed"
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "ai_predictions.csv")
    
    input_df['batch_id'] = batch_id
    input_df['farm_id'] = farm_id
    input_df['technician_id'] = technician_id
    input_df['ai_predicted_weight'] = prediction
    input_df['timestamp'] = datetime.datetime.now()
    
    if os.path.exists(csv_path):
        input_df.to_csv(csv_path, mode='a', header=False, index=False)
    else:
        input_df.to_csv(csv_path, index=False)
        
    st.info("Log de predição gravado.")

# -----------------------------------------------------------------------------
# 8. Signature
# -----------------------------------------------------------------------------
st.markdown("<br><br><br>", unsafe_allow_html=True)
st.markdown("<div style='text-align: center; color: gray;'><small>Desenvolvido por <b>Arleu Junior</b></small></div>", unsafe_allow_html=True)