# streamlit run C:\DataScience_Projects\vineetha_portfolio\vin_app.py

import streamlit as st
from PIL import Image
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'Vineetha_Gomathy_Resume.pdf')


# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Vineetha Gomathy - Aspiring Data Scientist Portfolio",
    page_icon="🧠",
    layout="wide"
)

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    #st.image("images/vineetha.jpg", width=150)  # Optional: your photo
    st.title("Vineetha Gomathy")
    st.markdown("**Aspiring Data Scientist | Former DB Architect**")
    st.markdown("📫 vineetha.inapp@gmail.com")
    #st.markdown("🌐 [GitHub](https://github.com/yourusername)")
    st.markdown("🔗 [LinkedIn](https://www.linkedin.com/in/vineetha-gomathy-21b506157)")
    with open(MODEL_PATH, "rb") as file:
        st.download_button("📄 Download Resume", file, "Vineetha_Gomathy_Resume.pdf")

# -----------------------------
# About Section
# -----------------------------
st.title("👩‍💼 About Me")
st.write("""
I'm a Database Architect with over 15 years of experience in PostgreSQL, SQL Server, Oracle, MariaDB and ETL pipelines.
Recently, I transitioned into **Data Science** and have built models in areas like:
- Face recognition using Haar cascades & wavelets
- Airbnb price prediction using XGBoost

""")
# -----------------------------
# Contact Section (Visible at Top)
# -----------------------------
st.header("📬 Contact Me")


st.write("vineetha.inapp@gmail.com")

st.write("📍 Based in Kerala, India")
st.markdown("[🔗 LinkedIn](https://www.linkedin.com/in/vineetha-gomathy-21b506157)")

st.info("💼 Currently looking for remote Data Science opportunities.")

# -----------------------------
# Skills Section
# -----------------------------
st.header("🧰 Skills & Tools")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Languages**")
    st.markdown("- Python")
    st.markdown("- SQL")
    st.markdown("- PL/SQL")

with col2:
    st.markdown("**Data Science**")
    st.markdown("- Pandas, NumPy, scikit-learn, XGBoost,Matplotlib, Seaborn")
    st.markdown("- NLP (Text Preprocessing, TF-IDF, Word Embeddings)")
    st.markdown("- Deep Learning (Keras, TensorFlow – basics)")

with col3:
    st.markdown("**Tools & DBs**")
    st.markdown("- AWS RDS-PostgreSQL, MariaDB, Oracle, SQL Server")
    st.markdown("- Streamlit, Flask")
    st.markdown("- Git, VS Code")


# -----------------------------
# Projects Section
# -----------------------------
st.header("📊 Data Science Projects")

st.subheader("1. Airbnb Price Prediction")
st.write("""
A regression model predicting Airbnb listing prices using XGBoost. Includes:
- Feature engineering
- Log transformation of target variable
- SHAP explanations
""")
#st.markdown("[🔗 View Code](https://github.com/yourusername/airbnb-price-prediction)")

st.subheader("2. Face Recognition Model")
st.write("""
Built using Haar cascades for face & eye detection, wavelet transform for feature extraction, and multiple classifiers.
Selected Logistic Regression based on performance.
""")
#st.markdown("[🔗 View Code](https://github.com/yourusername/face-recognition-model)")

#st.subheader("3. Diabetes Prediction App")
#st.write("""
#A classification model trained on Pima Indian dataset, with a live Streamlit app for testing diabetes risk.
#""")
#st.markdown("[🔗 View Code](https://github.com/yourusername/diabetes-prediction-app)")
# -----------------------------
# Data Engineering & Analytics Projects
# -----------------------------
st.header("📦 Data Engineering & Analytics Projects")

st.subheader("🔹 iFly Loyalty Data Migration – Solaseed Airlines, Japan")
st.write("""
**Role:** Database Architect  
**Company:** IBS Software, Kochi  
**Duration:** Dec 2023 – Sep 2024  
**Tech Stack:** Python, PostgreSQL (AWS RDS)

- Collaborated with business analysts and stakeholders to analyze source system data and map it to the iFly Loyalty data model.
- Built Python-based data cleansing and transformation scripts to ensure high-quality, structured input for the iFly system.
- Led the data migration process for Solaseed Airlines (a Japan Airlines partner), aligning legacy data with modern relational schemas to enable seamless integration and reporting.
- Focused on data quality checks, null handling, and load validation, preparing clean datasets for loyalty-based analytics.
""")

st.subheader("🔹 CASS 2.0 – IATA (International Air Transport Association)")
st.write("""
**Role:** Database Architect  
**Company:** IBS Software, Kochi  
**Duration:** Sep 2021 – Nov 2023  
**Tech Stack:** PostgreSQL (AWS RDS)

- Designed and maintained relational schemas supporting IATA’s global Cargo Accounts Settlement System, used by over 115 airlines.
- Worked cross-functionally to interpret functional and system requirements into scalable data models.
- Led data migration, performed validation and performance tuning, and supported QA and production teams in live data flow monitoring and issue resolution.
- Enabled downstream analytics by ensuring data consistency and traceability for airline settlement records.
""")

st.subheader("🔹 Paragon Meter Tracking Platform – Vantage Meters, UK")
st.write("""
**Role:** Database Consultant  
**Company:** Emvigo Technologies, Kochi  
**Duration:** Aug 2020 – Aug 2021  
**Tech Stack:** MariaDB (AWS RDS), Pentaho

- Led a team of DB developers in building and optimizing data pipelines for Paragon — a SaaS platform for metering asset lifecycle tracking.
- Designed multi-tenant database schemas and collaborated with business teams for data mapping and validation.
- Used Pentaho ETL tools to automate data ingestion, transformation, and error handling from multiple energy providers.
- Enabled structured data capture to support future integration with predictive maintenance and asset analytics modules.
""")

st.subheader("🔹 AXIOM HQ – Axiom, USA")
st.write("""
**Role:** Database Consultant  
**Company:** InApp Technologies, Trivandrum  
**Duration:** Jul 2016 – Mar 2019  
**Tech Stack:** SQL Server 2012/2014

- Supported the development of a scalable analytics-driven ERP platform offering modules like General Ledger, Inventory, and Order Processing for distributors in the US.
- Designed and optimized relational database schemas for high-volume modules such as Accounts Receivable, Order Management, and Inventory Control, ensuring data consistency and scalability.
- Developed complex SQL procedures and business logic to support real-time dashboards, reporting modules, and transactional insights.
- Worked closely with product managers and QA teams to validate data integrity and business rule compliance, laying the foundation for future analytics and BI integration.
- Contributed to data extraction and preprocessing logic for financial and operational datasets used by Axiom's clients for decision-making.
""")

st.subheader("🔹 Ultrasound Documentation & Reporting System – NetReach Software, Singapore")
st.write("""
**Role:** Database Developer  
**Company:** NetReach Software, Singapore  
**Duration:** Nov 2003 – Nov 2004  
**Tech Stack:** Oracle 9i, PL/SQL

- Developed backend logic for a healthcare imaging system designed to analyze and report fetal ultrasound results.
- Created PL/SQL procedures, functions, and triggers to automate sonographic calculations, including fetal age estimation, graphing metrics, and measurement summaries.
- Structured clinical imaging data to support downstream reporting, visualization, and eventual integration with diagnostic dashboards.
- Focused on ensuring accuracy, completeness, and auditability of healthcare data — critical for diagnostic reliability and compliance.
- Enabled structured output generation for sonographers and radiologists, supporting clinical decision-making.
""")


# -----------------------------
# References Section
# -----------------------------
st.header("💬 References")

st.write("""
**Suchithra Vinod**  
Vice President, InApp Technologies, Technopark, Trivandrum  
[LinkedIn Profile](https://www.linkedin.com/in/suchi-vinod-90293018/)

**Resmi Raman**  
Service Delivery Manager, IBS Software, Kochi  
[LinkedIn Profile](https://www.linkedin.com/in/resmi-raman-0761367/)
""")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.write("© 2025 Vineetha Gomathy")
