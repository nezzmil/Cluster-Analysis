import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.metrics import silhouette_score

st.title("Global Development Dataset - Cluster Analysis")

# Sidebar for user inputs
st.sidebar.title("Data Upload")
uploaded_file = st.sidebar.file_uploader("Upload your Excel file", type=["xlsx"])

if uploaded_file is not None:
    # Load the dataset
    sheet_name = st.sidebar.text_input("Enter sheet name", "world_development")
    df = pd.read_excel(uploaded_file, sheet_name=sheet_name)
    st.write("### Dataset Overview", df.head())


        # Convert percentage columns to numeric after removing the '%'sign,if applicable
    for column in df.columns:
        if df[column].dtype == 'object' and df[column].str.contains('%').any():
            df[column] = df[column].str.rstrip('%').astype(float)
    
    # Select only numeric columns for filling NaN values
    numeric_cols = df.select_dtypes(include=['number']).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    # Display the updated DataFrame
    st.write("Updated DataFrame:")
    st.write(df)

    # Data Cleaning and Preprocessing
    st.write("### Data Cleaning")
    for col in ['GDP', 'Health Exp/Capita', 'Tourism Inbound', 'Tourism Outbound']:
        df[col] = df[col].str.replace('$', '').str.replace(',', '').astype(float)
    numeric_cols = df.select_dtypes(include=['number']).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    # Fill non-numeric columns with their mode
    non_numeric_cols = df.select_dtypes(exclude=['number']).columns
    for col in non_numeric_cols:
       df[col].fillna(df[col].mode()[0], inplace=True)


    st.write("Cleaned Data", df.head())

    # Data Visualization
    st.write("### Data Distribution")
    st.write("Select a column to visualize")
    selected_column = st.selectbox("Choose a column", df.columns)
    fig, ax = plt.subplots()
    sns.histplot(df[selected_column], kde=True, ax=ax)
    st.pyplot(fig)

    # Clustering Analysis
    st.write("### Clustering Analysis")
    df_features = df.select_dtypes(include=['float64', 'int64'])
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_features)

    # K-Means
    st.write("#### K-Means Clustering")  
    inertia = []
    k_values = range(1, 11)

    for k in k_values:
    	kmeans = KMeans(n_clusters=k, random_state=42)
    	kmeans.fit(scaled_data)
    	inertia.append(kmeans.inertia_)
    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(k_values, inertia, marker='o', linestyle='--')
    ax.set_xlabel('Number of Clusters (k)')
    ax.set_ylabel('Inertia')
    ax.set_title('Elbow Method for Optimal k')
    st.pyplot(fig)
    k = st.slider("Select number of clusters (K)", 2, 10, 4)
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_data)
    labels_kmeans = kmeans.labels_
    silhouette_kmeans = silhouette_score(scaled_data, labels_kmeans)
    st.write(f"Silhouette Score for K-Means: {silhouette_kmeans:.2f}")

    # Hierarchical Clustering
    st.write("#### Hierarchical Clustering")
    linked = linkage(scaled_data, method='ward')
    fig, ax = plt.subplots(figsize=(10, 7))
    dendrogram(linked, ax=ax)
    st.pyplot(fig)
    num_clusters = st.slider("Select number of clusters for Hierarchical", 2, 10, 4)
    clusters_hierarchical = fcluster(linked, num_clusters, criterion='maxclust')
    silhouette_hierarchical = silhouette_score(scaled_data, clusters_hierarchical)
    st.write(f"Silhouette Score for Hierarchical: {silhouette_hierarchical:.2f}")

    # DBSCAN
    st.write("#### DBSCAN Clustering")
    eps = st.slider("Select epsilon (eps)", 0.1, 2.0, 0.5)
    min_samples = st.slider("Select minimum samples", 1, 10, 5)
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels_dbscan = dbscan.fit_predict(scaled_data)
    silhouette_dbscan = silhouette_score(scaled_data, labels_dbscan) if len(set(labels_dbscan)) > 1 else "N/A"
    st.write(f"Silhouette Score for DBSCAN: {silhouette_dbscan}")

    # Agglomerative Clustering
    st.write("#### Agglomerative Clustering")
    agg = AgglomerativeClustering(n_clusters=num_clusters)
    labels_agglo = agg.fit_predict(scaled_data)
    silhouette_agglo = silhouette_score(scaled_data, labels_agglo)
    st.write(f"Silhouette Score for Agglomerative: {silhouette_agglo:.2f}")

    # Visualization of Clusters
    st.write("### Cluster Visualization")
    fig, ax = plt.subplots()
    sns.scatterplot(x=scaled_data[:, 0], y=scaled_data[:, 1], hue=labels_kmeans, palette="viridis", ax=ax)
    ax.set_title("K-Means Clustering")
    st.pyplot(fig)

else:
    st.write("Please upload an Excel file to proceed.")