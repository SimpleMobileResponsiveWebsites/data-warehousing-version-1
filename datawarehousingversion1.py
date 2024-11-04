import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import json
import hashlib
import os

# Directory to store data sources and metadata
DATA_DIR = "data_sources"
METADATA_FILE = os.path.join(DATA_DIR, "data_sources_metadata.csv")

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

# Initialize metadata file if not present
if not os.path.exists(METADATA_FILE):
    pd.DataFrame(columns=["id", "name", "type", "schema", "created_at", "last_updated"]).to_csv(METADATA_FILE, index=False)

# Data ingestion function using CSV
def upload_data():
    st.header("Data Ingestion")
    uploaded_file = st.file_uploader("Upload Data File", type=['csv', 'xlsx', 'json'])
    
    if uploaded_file is not None:
        try:
            # Determine file type and load data into DataFrame
            file_type = uploaded_file.name.split('.')[-1]
            if file_type == 'csv':
                df = pd.read_csv(uploaded_file)
            elif file_type == 'xlsx':
                df = pd.read_excel(uploaded_file)
            elif file_type == 'json':
                df = pd.read_json(uploaded_file)

            # Generate source ID and save metadata
            source_id = hashlib.md5(uploaded_file.name.encode()).hexdigest()
            schema = json.dumps(df.dtypes.astype(str).to_dict())
            metadata = {
                "id": source_id,
                "name": uploaded_file.name,
                "type": file_type,
                "schema": schema,
                "created_at": pd.Timestamp.now(),
                "last_updated": pd.Timestamp.now()
            }

            # Append metadata to the metadata file
            metadata_df = pd.concat([pd.read_csv(METADATA_FILE), pd.DataFrame([metadata])], ignore_index=True)
            metadata_df.to_csv(METADATA_FILE, index=False)

            # Save the data to a .csv file in the data directory
            data_file_path = os.path.join(DATA_DIR, f"{source_id}.csv")
            df.to_csv(data_file_path, index=False)

            st.success(f"Successfully ingested {len(df)} records from {uploaded_file.name}")

        except Exception as e:
            st.error(f"Error during data ingestion: {str(e)}")

# Data transformation function with CSV
def transform_data():
    st.header("Data Transformation")
    metadata_df = pd.read_csv(METADATA_FILE)
    sources = dict(zip(metadata_df["id"], metadata_df["name"]))
    
    if not sources:
        st.warning("No data sources available. Please upload data first.")
        return

    # Select data source
    selected_source = st.selectbox("Select Data Source", list(sources.values()))
    source_id = [k for k, v in sources.items() if v == selected_source][0]
    data_file_path = os.path.join(DATA_DIR, f"{source_id}.csv")

    # Load data for transformation
    df = pd.read_csv(data_file_path)

    # Transformation options
    transformation_options = ["Filter by column", "Sort by column", "Remove duplicates"]
    selected_transformation = st.selectbox("Select Transformation", transformation_options)

    if selected_transformation == "Filter by column":
        columns = df.columns.tolist()
        selected_column = st.selectbox("Select Column", columns)
        filter_value = st.text_input("Filter Value")
        df = df[df[selected_column] == filter_value]
    elif selected_transformation == "Sort by column":
        columns = df.columns.tolist()
        selected_column = st.selectbox("Select Column", columns)
        df = df.sort_values(by=selected_column)
    elif selected_transformation == "Remove duplicates":
        df = df.drop_duplicates()

    # Save the transformed data
    df.to_csv(data_file_path, index=False)

    st.success("Data transformed successfully!")

# Data export function
def export_data():
    st.header("Data Export")
    metadata_df = pd.read_csv(METADATA_FILE)
    sources = dict(zip(metadata_df["id"], metadata_df["name"]))
    
    if not sources:
        st.warning("No data sources available. Please upload data first.")
        return
    
    # Select source to export
    selected_source = st.selectbox("Select Data Source to Export", list(sources.values()))
    source_id = [k for k, v in sources.items() if v == selected_source][0]
    data_file_path = os.path.join(DATA_DIR, f"{source_id}.csv")

    # Load data for export
    df = pd.read_csv(data_file_path)

    # Export options
    export_options = ["Export to CSV", "Export to JSON", "Visualize data"]
    selected_export = st.selectbox("Select Export Option", export_options)

    if selected_export == "Export to CSV":
        df.to_csv("export
