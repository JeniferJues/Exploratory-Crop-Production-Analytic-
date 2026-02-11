import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(
    page_title="Malaysia Agriculture Analytics",
    layout="wide"
)

# --------------------------------------------------
# Load Data
# --------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("data/final_crop.csv")

@st.cache_data
def load_clusters():
    return pd.read_csv("data/state_clusters.csv")

@st.cache_resource
def load_model():
    return joblib.load("model/best_model.pkl")

df = load_data()
cluster_df = load_clusters()
model = load_model()

# --------------------------------------------------
# Header
# --------------------------------------------------
st.image("assets/Header.png", use_column_width=True)
st.title("Malaysia Crop Production Analytics")
st.markdown(
    "Exploratory Analysis • State Clustering • Machine Learning Prediction"
)

# --------------------------------------------------
# Sidebar Navigation
# --------------------------------------------------
page = st.sidebar.radio(
    "Navigate",
    [
        "📊 Power BI Dashboard",
        "📈 Exploratory Data Analysis",
        "🧠 Cluster Insights",
        "🤖 Production Prediction"
    ]
)

# ==================================================
# PAGE 1 – Power BI
# ==================================================
if page == "📊 Power BI Dashboard":

    st.header("Interactive Power BI Dashboard")

    power_bi_url = "https://app.powerbi.com/view?r=eyJrIjoiYWYzOTk5NjctNTI4Mi00MGU3LTg1Y2MtYjY3YTg4YzNlNGY0IiwidCI6ImE2M2JiMWE5LTQ4YzItNDQ4Yi04NjkzLTMzMTdiMDBjYTdmYiIsImMiOjEwfQ%3D%3D"

    st.components.v1.iframe(
        power_bi_url,
        width=1200,
        height=800
    )

# ==================================================
# PAGE 2 – EDA
# ==================================================
elif page == "📈 Exploratory Data Analysis":

    st.header("Production Trend Analysis")

    crop_selection = st.selectbox(
        "Select Crop Type",
        sorted(df["crop_type"].unique())
    )

    filtered = df[df["crop_type"] == crop_selection]

    # Use log scale for visualization (because skewed)
    filtered["log_production"] = np.log10(filtered["production"] + 1)

    fig = px.line(
        filtered,
        x="year",
        y="log_production",
        color="state",
        title=f"{crop_selection} Production Trend (Log Scale)"
    )

    st.plotly_chart(fig, use_container_width=True)

    # Distribution
    st.subheader("Distribution of Total Production by State")

    state_total = df.groupby("state")["production"].sum().reset_index()
    state_total["log_total"] = np.log10(state_total["production"] + 1)

    fig2 = px.histogram(
        state_total,
        x="log_total",
        nbins=10,
        title="Distribution of Total Production (Log Scale)"
    )

    st.plotly_chart(fig2, use_container_width=True)

    st.caption("Log scale used due to high production skewness (billions vs thousands).")

# ==================================================
# PAGE 3 – CLUSTERING
# ==================================================
elif page == "🧠 Cluster Insights":

    st.header("State Clustering by Agricultural Profile")

    fig = px.scatter(
        cluster_df,
        x="avg_production",
        y="growth_rate",
        color="cluster_label",
        size="variability",
        hover_name="state",
        title="Clustered States by Production & Growth"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    ### Cluster Interpretation
    - **Cluster 0**: Flower Specialized  
    - **Cluster 1**: Mixed Crop  
    - **Cluster 2**: Flower + Rice focused
    """)

# ==================================================
# PAGE 4 – ML PREDICTION
# ==================================================
elif page == "🤖 Production Prediction":

    st.header("Crop Production Prediction (Random Forest Model)")

    st.markdown("""
    Model trained using log-transformed production values  
    Final model selected based on lowest RMSE and highest R².
    """)

    col1, col2, col3 = st.columns(3)

    with col1:
        state = st.selectbox("State", sorted(df["state"].unique()))

    with col2:
        crop = st.selectbox("Crop Type", sorted(df["crop_type"].unique()))

    with col3:
        year = st.slider("Year", int(df["year"].min()), int(df["year"].max()+3), 2023)

    if st.button("Predict Production"):

        input_df = pd.DataFrame({
            "state": [state],
            "crop_type": [crop],
            "year": [year]
        })

        # Model prediction (log scale output)
        log_prediction = model.predict(input_df)

        # Reverse log transformation
        prediction = np.expm1(log_prediction[0])

        st.success(f"Predicted Production: {prediction:,.2f}")

        st.caption("Prediction converted back from log scale to actual production value.")
