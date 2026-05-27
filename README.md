
# AI Poultry Weight Predictor & Bias Correction

> Offline-first toolkit to ingest farm data, predict broiler slaughter weight, and deliver actionable insights through a Streamlit web app for supervisors and a Power BI dashboard for managers.

## Business Context
In poultry production, logistics depend on accurate weight estimation. Slaughterhouses require flock weight predictions within a **±100g margin of error**. Human bias and biological factors (fasting, transport shrinkage) often lead to deviations, creating operational bottlenecks and financial losses.

## Solution
This project implements a complete Data Science pipeline to reduce farmer bias and improve decision-making:

1. **Data Engineering (DuckDB):** Historical data transformed into a Star Schema for efficient analytical queries.  
2. **Machine Learning (XGBoost):** Regression model trained to capture farmer bias, growth curves, and transport shrinkage.  
3. **Operational Front-End (Streamlit):** Local web app for supervisors to input daily data and instantly receive AI-corrected predictions.  
4. **Executive Dashboard (Power BI):** BI layer for managers to monitor AI accuracy, bias trends, and logistics KPIs.

## Tech Stack
- **Language:** Python 3  
- **Data Warehouse:** DuckDB  
- **Machine Learning:** Scikit-Learn, XGBoost, Pandas  
- **Web Interface:** Streamlit  
- **Business Intelligence:** Power BI  

## Project Structure
```text
poultry-weight-prediction/
├── data/               # Raw, input, and processed datasets
├── models/             # Serialized ML models (.joblib)
├── notebooks/          # EDA and feature engineering
├── src/                # ETL and ML scripts
├── app.py              # Streamlit web application
└── README.md
```

## How to Run
1. Clone the repository and install dependencies:
   ```bash
   git clone https://github.com/arleujr/poultry-weight-prediction.git
   cd poultry-weight-prediction
   python -m venv .venv
   source .venv/Scripts/activate  # On Windows
   pip install -r requirements.txt
   ```

2. Build the Data Warehouse & Train the Model:
   ```bash
   python src/data/transform.py
   python src/models/train.py
   ```

3. Launch the Supervisor Web App:
   ```bash
   streamlit run app.py
  ```


**Author: Arleu Junior**  
Agronomy student at UFV, bridging IoT and Data Science to deliver end-to-end solutions that connect agriculture and technology.


