import streamlit as st
import pandas as pd
import joblib
import datetime

# 1. Page Configuration (UI/UX)
st.set_page_config(
    page_title="Estimativa de Peso de Abate",
    page_icon="🐔",
    layout="centered"
)

# 2. Load the AI Model (Cached for performance)
@st.cache_resource
def load_model():
    model = joblib.load("models/xgb_weight_predictor.joblib")
    expected_columns = joblib.load("models/model_features.joblib")
    return model, expected_columns

try:
    model, expected_columns = load_model()
except FileNotFoundError:
    st.error("Arquivos do modelo não encontrados. Treine o modelo primeiro.")
    st.stop()

# 3. Static Farm Database (Business Logic: Farm location and shed type are fixed)
# In a real scenario, this would come from an SQL database (dim_farm).
FARM_DB = {
    "GRANJA_01": {"distancia_km": 15.5, "tipo_galpao": "Convencional"},
    "GRANJA_05": {"distancia_km": 45.5, "tipo_galpao": "Dark House"},
    "GRANJA_12": {"distancia_km": 80.0, "tipo_galpao": "Dark House"},
}

# 4. Front-end Header (Portuguese for the end-user)
st.title("Estimativa do Peso de abate")
st.markdown("Insira os dados do lote para estimar o peso final na balança do frigorífico.")
st.divider()

# 5. User Input Form
with st.form("prediction_form"):
    st.subheader("Detalhes Operacionais do Lote")
    
    col1, col2 = st.columns(2)
    
    with col1:
        batch_id = st.text_input("ID do Lote", "LOTE_HOJE_002")
        farm_id = st.selectbox("ID da Granja", list(FARM_DB.keys()))
        farmer_id = st.selectbox("ID do Granjeiro", ["GRANJEIRO_06", "GRANJEIRO_12", "GRANJEIRO_15"])
        placement_date = st.date_input("Data de Alojamento", datetime.date(2026, 4, 1))
        slaughter_date = st.date_input("Data Prevista de Abate", datetime.date(2026, 5, 15))
        strain = st.selectbox("Linhagem", ["Ross 308", "Cobb 500"])
        flock_sex = st.selectbox("Sexo do Lote", ["Macho", "Fêmea", "Misto"])
        
    with col2:
        # Display static info for the user to see, but without allowing editing
        st.info(f"📍 Distância: {FARM_DB[farm_id]['distancia_km']} km | Galpão: {FARM_DB[farm_id]['tipo_galpao']}")
        
        fasting_time = st.number_input("Tempo de Jejum (horas)", value=8)
        mortality_rate = st.number_input("Mortalidade Acumulada (%)", value=2.1, step=0.1)
        weight_30d = st.number_input("Peso Amostral aos 30 dias (g)", value=1650)
        weight_40d = st.number_input("Peso Amostral aos 40 dias (g)", value=2550)
        farmer_estimated_weight = st.number_input("Estimativa do Granjeiro (g)", value=3200)

    # Submit Button
    submitted = st.form_submit_button("Calcular Estimativa", type="primary")

# 6. Inference Logic (Runs when button is clicked)
if submitted:
    # Calculate slaughter age dynamically
    slaughter_age_days = (slaughter_date - placement_date).days
    
    # Map inputs to the exact column names expected by the AI model
    input_dict = {
        'weight_30d_g': [weight_30d],
        'weight_40d_g': [weight_40d],
        'farmer_estimated_weight_g': [farmer_estimated_weight],
        'flock_sex': [flock_sex],
        'fasting_time_hrs': [fasting_time],
        'mortality_rate_pct': [mortality_rate],
        'transport_distance_km': [FARM_DB[farm_id]['distancia_km']], # Pulled from the static DB
        'shed_type': [FARM_DB[farm_id]['tipo_galpao']],             # Pulled from the static DB
        'slaughter_age_days': [slaughter_age_days]
    }
    
    input_df = pd.DataFrame(input_dict)
    
    # Preprocess categorical variables
    features_encoded = pd.get_dummies(input_df)
    features_aligned = features_encoded.reindex(columns=expected_columns, fill_value=0)
    
    # Generate Prediction
    prediction = model.predict(features_aligned)[0]
    correction = farmer_estimated_weight - prediction
    
    # 7. Display Results (UI Output)
    st.divider()
    st.subheader("📊 Resultado da Previsão")
    
    res_col1, res_col2 = st.columns(2)
    
    res_col1.metric(
        label="Estimativa Humana", 
        value=f"{farmer_estimated_weight:,.0f} g"
    )
    
    res_col2.metric(
        label="Previsão Final da IA", 
        value=f"{prediction:,.2f} g",
        delta=f"{correction:,.2f} g (Correção de Viés)",
        delta_color="inverse"
    )
    
    # Business logic alerts
    if abs(correction) > 100:
        st.warning(f"⚠️ **Atenção:** Alto viés humano detectado. A estimativa do granjeiro errou por {abs(correction):.0f}g. Utilize a previsão da IA para o planejamento do abatedouro.")
    else:
        st.success("✅ A estimativa do granjeiro está excelente e dentro da margem operacional de 100g.")
        # 8. Footer / Assinatura
st.markdown("<br><br>", unsafe_allow_html=True) # Dá um espaço visual para respirar
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        <small>
            Desenvolvido por <b>Arleu Junior</b> 
                </div>
    """, 
    unsafe_allow_html=True
)