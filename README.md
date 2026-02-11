###🔹 Project Overview 
This project analyzes Malaysia’s crop production patterns across states and years using exploratory data analysis, machine learning, and interactive dashboards. The goal is to uncover regional crop specialization, production trends, and risk patterns, and translate these insights into decision-ready visuals.

Tools & Tech
•	Python (Pandas, NumPy, Scikit-learn, Plotly)

•	Power BI (Dashboard storytelling)

•	Streamlit (Interactive deployment)

•	GeoJSON (Malaysia state mapping)

•	Data source: DOSM Open Data (Agriculture – Crop Production)
___________________________________________________________________________________________________________________________________________________________________________________________
🔹 Data Preparation & EDA (Python)
Exploratory data analysis was conducted in Python to understand distribution, trends, and regional variation.
Key steps:
•	Data cleaning and aggregation by state, crop type, and year
•	Trend analysis (2017–2022)
•	Production distribution analysis across states
•	Log-scaling for highly skewed production values
_______________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________
🔹 Machine Learning: Pattern Discovery
Unsupervised Learning – K-Means Clustering
States were clustered based on:
•	Crop mix proportions
•	Production volume
•	Growth trends
This allowed discovery of natural groupings without predefined labels.
📊 Model Justification
•	Elbow Method → optimal number of clusters
•	Silhouette Score → cluster separation quality
•	PCA → 2D visualization for interpretability
🏷 Cluster Interpretation
Clusters were labeled using dominant crop composition:
•	Flower-specialized
•	Mixed crop (flowers + paddy)
•	Flower + rice-focused
📌 ML output was exported and reused in Power BI and Streamlit.
_________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________
🔹 Power BI Dashboard (Storytelling Layer)
Power BI was used as the decision-facing layer to translate technical findings into intuitive insights.
