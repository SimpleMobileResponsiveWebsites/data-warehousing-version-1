import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sqlite3
import json
import io
from pathlib import Path
import hashlib

# Configure Streamlit page
st.set_page_config(
    page_title="Data Warehouse Management System",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize database
def init_db():
    conn = sqlite3.connect('data_warehouse.db')
    c = conn.cursor()
    
    # Create tables for metadata management
    c.execute('''CREATE TABLE IF NOT EXISTS data_sources
                 (id TEXT PRIMARY KEY, 
                  name TEXT, 
                  type TEXT,
                  schema TEXT,
                  created_at TIMESTAMP,
                  last_updated TIMESTAMP)''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS etl_jobs
                 (id TEXT PRIMARY KEY,
                  source_id TEXT,
                  status TEXT,
                  start_time TIMESTAMP,
                  end_time TIMESTAMP,
                  records_processed INTEGER,
                  error_message TEXT)''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS data_quality_rules
                 (id TEXT PRIMARY KEY,
                  source_id TEXT,
                  rule_type TEXT,
                  rule_definition TEXT,
                  is_active BOOLEAN)''')
    
    conn.commit()
    return conn

# Data ingestion functions
def upload_data():
    st.header("Data Ingestion")
    
    # File upload
    uploaded_file = st.file_uploader("Upload Data File", type=['csv', 'xlsx', 'json'])
    if uploaded_file is not None:
        try:
            # Determine file type and read accordingly
            file_type = uploaded_file.name.split('.')[-1]
            if file_type == 'csv':
                df = pd.read_csv(uploaded_file)
            elif file_type == 'xlsx':
                df = pd.read_excel(uploaded_file)
            elif file_type == 'json':
                df = pd.read_json(uploaded_file)
            
            # Generate source ID
            source_id = hashlib.md5(uploaded_file.name.encode()).hexdigest()
            
            # Store metadata
            conn = init_db()
            c = conn.cursor()
            c.execute('''INSERT OR REPLACE INTO data_sources 
                        (id, name, type, schema, created_at, last_updated)
                        VALUES (?, ?, ?, ?, ?, ?)''',
                     (source_id, uploaded_file.name, file_type,
                      json.dumps(df.dtypes.astype(str).to_dict()),
                      datetime.now(), datetime.now()))
            
            conn.commit()
            
            # Save data to warehouse
            df.to_sql(f'source_{source_id}', conn, if_exists='replace', index=False)
            
            st.success(f"Successfully ingested {len(df)} records from {uploaded_file.name}")
            
            # Display data preview
            st.subheader("Data Preview")
            st.dataframe(df.head())
            
            # Display basic statistics
            st.subheader("Data Statistics")
            col1, col2 = st.columns(2)
            with col1:
                st.write("Numeric Columns Summary:")
                st.dataframe(df.describe())
            with col2:
                st.write("Missing Values Summary:")
                st.dataframe(df.isnull().sum().to_frame(name='Missing Count'))
            
        except Exception as e:
            st.error(f"Error during data ingestion: {str(e)}")

# Data transformation functions
def transform_data():
    st.header("Data Transformation")
    
    # Get available data sources
    conn = init_db()
    c = conn.cursor()
    c.execute("SELECT id, name FROM data_sources")
    sources = dict(c.fetchall())
    
    if not sources:
        st.warning("No data sources available. Please upload data first.")
        return
    
    # Select source
    selected_source = st.selectbox("Select Data Source", list(sources.values()))
    source_id = [k for k, v in sources.items() if v == selected_source][0]
    
    # Load data
    df = pd.read_sql(f"SELECT * FROM source_{source_id}", conn)
    
    # Transformation options
    st.subheader("Available Transformations")
    
    transform_type = st.selectbox("Select Transformation Type", 
                                ["Column Operations", "Filtering", "Aggregation"])
    
    if transform_type == "Column Operations":
        col_operation = st.selectbox("Select Operation", 
                                   ["Rename Columns", "Change Data Type", "Add Calculated Column"])
        
        if col_operation == "Rename Columns":
            cols = df.columns.tolist()
            col_mapping = {}
            st.write("Enter new names for columns:")
            for col in cols:
                new_name = st.text_input(f"New name for {col}", col)
                col_mapping[col] = new_name
            
            if st.button("Apply Renaming"):
                df = df.rename(columns=col_mapping)
                st.success("Columns renamed successfully")
                st.dataframe(df.head())
        
        elif col_operation == "Change Data Type":
            col = st.selectbox("Select Column", df.columns)
            new_type = st.selectbox("Select New Type", ["int", "float", "str", "datetime"])
            
            if st.button("Change Type"):
                try:
                    df[col] = df[col].astype(new_type)
                    st.success(f"Changed type of {col} to {new_type}")
                    st.dataframe(df.head())
                except Exception as e:
                    st.error(f"Error changing type: {str(e)}")
        
        elif col_operation == "Add Calculated Column":
            col_name = st.text_input("New Column Name")
            available_cols = df.select_dtypes(include=[np.number]).columns
            col1 = st.selectbox("Select First Column", available_cols)
            operation = st.selectbox("Select Operation", ["+", "-", "*", "/"])
            col2 = st.selectbox("Select Second Column", available_cols)
            
            if st.button("Add Column"):
                try:
                    if operation == "+":
                        df[col_name] = df[col1] + df[col2]
                    elif operation == "-":
                        df[col_name] = df[col1] - df[col2]
                    elif operation == "*":
                        df[col_name] = df[col1] * df[col2]
                    elif operation == "/":
                        df[col_name] = df[col1] / df[col2]
                    st.success(f"Added new column {col_name}")
                    st.dataframe(df.head())
                except Exception as e:
                    st.error(f"Error adding column: {str(e)}")
    
    elif transform_type == "Filtering":
        col = st.selectbox("Select Column for Filtering", df.columns)
        operation = st.selectbox("Select Filter Operation", 
                               ["equals", "greater than", "less than", "contains"])
        filter_value = st.text_input("Enter Filter Value")
        
        if st.button("Apply Filter"):
            try:
                if operation == "equals":
                    df = df[df[col] == filter_value]
                elif operation == "greater than":
                    df = df[df[col] > float(filter_value)]
                elif operation == "less than":
                    df = df[df[col] < float(filter_value)]
                elif operation == "contains":
                    df = df[df[col].astype(str).str.contains(filter_value)]
                st.success("Filter applied successfully")
                st.dataframe(df.head())
            except Exception as e:
                st.error(f"Error applying filter: {str(e)}")
    
    elif transform_type == "Aggregation":
        group_cols = st.multiselect("Select Columns to Group By", df.columns)
        agg_col = st.selectbox("Select Column to Aggregate", df.select_dtypes(include=[np.number]).columns)
        agg_func = st.selectbox("Select Aggregation Function", ["sum", "mean", "count", "min", "max"])
        
        if st.button("Apply Aggregation"):
            try:
                df_agg = df.groupby(group_cols)[agg_col].agg(agg_func).reset_index()
                st.success("Aggregation applied successfully")
                st.dataframe(df_agg)
            except Exception as e:
                st.error(f"Error applying aggregation: {str(e)}")

# Data quality functions
def manage_data_quality():
    st.header("Data Quality Management")
    
    conn = init_db()
    c = conn.cursor()
    
    # Get available data sources
    c.execute("SELECT id, name FROM data_sources")
    sources = dict(c.fetchall())
    
    if not sources:
        st.warning("No data sources available. Please upload data first.")
        return
    
    # Select source
    selected_source = st.selectbox("Select Data Source", list(sources.values()))
    source_id = [k for k, v in sources.items() if v == selected_source][0]
    
    # Data quality rule management
    st.subheader("Data Quality Rules")
    
    # Add new rule
    with st.expander("Add New Rule"):
        rule_type = st.selectbox("Rule Type", 
                               ["Completeness", "Accuracy", "Consistency", "Validity"])
        
        if rule_type == "Completeness":
            col = st.selectbox("Select Column", 
                             pd.read_sql(f"SELECT * FROM source_{source_id}", conn).columns)
            threshold = st.slider("Missing Values Threshold (%)", 0, 100, 5)
            rule_def = json.dumps({"column": col, "threshold": threshold})
        
        elif rule_type == "Accuracy":
            col = st.selectbox("Select Column", 
                             pd.read_sql(f"SELECT * FROM source_{source_id}", conn).columns)
            min_val = st.number_input("Minimum Value")
            max_val = st.number_input("Maximum Value")
            rule_def = json.dumps({"column": col, "min": min_val, "max": max_val})
        
        elif rule_type == "Consistency":
            col1 = st.selectbox("Select First Column", 
                              pd.read_sql(f"SELECT * FROM source_{source_id}", conn).columns)
            col2 = st.selectbox("Select Second Column", 
                              pd.read_sql(f"SELECT * FROM source_{source_id}", conn).columns)
            relation = st.selectbox("Relation", ["equal", "greater than", "less than"])
            rule_def = json.dumps({"column1": col1, "column2": col2, "relation": relation})
        
        elif rule_type == "Validity":
            col = st.selectbox("Select Column", 
                             pd.read_sql(f"SELECT * FROM source_{source_id}", conn).columns)
            pattern = st.text_input("Regular Expression Pattern")
            rule_def = json.dumps({"column": col, "pattern": pattern})
        
        if st.button("Add Rule"):
            rule_id = hashlib.md5(f"{source_id}_{rule_type}_{rule_def}".encode()).hexdigest()
            c.execute('''INSERT OR REPLACE INTO data_quality_rules 
                        (id, source_id, rule_type, rule_definition, is_active)
                        VALUES (?, ?, ?, ?, ?)''',
                     (rule_id, source_id, rule_type, rule_def, True))
            conn.commit()
            st.success("Rule added successfully")
    
    # View and manage existing rules
    st.subheader("Existing Rules")
    c.execute('''SELECT id, rule_type, rule_definition, is_active 
                 FROM data_quality_rules WHERE source_id = ?''', (source_id,))
    rules = c.fetchall()
    
    if rules:
        for rule in rules:
            rule_id, rule_type, rule_def, is_active = rule
            with st.expander(f"{rule_type} Rule - {rule_id[:8]}"):
                st.write(f"Definition: {json.loads(rule_def)}")
                new_status = st.checkbox("Active", value=is_active, key=rule_id)
                
                if new_status != is_active:
                    c.execute('''UPDATE data_quality_rules 
                                SET is_active = ? WHERE id = ?''', (new_status, rule_id))
                    conn.commit()
                
                if st.button(f"Delete Rule {rule_id[:8]}"):
                    c.execute("DELETE FROM data_quality_rules WHERE id = ?", (rule_id,))
                    conn.commit()
                    st.success("Rule deleted successfully")
                    st.experimental_rerun()

# Analytics and reporting functions
def analyze_data():
    st.header("Analytics & Reporting")
    
    conn = init_db()
    c = conn.cursor()
    
    # Get available data sources
    c.execute("SELECT id, name FROM data_sources")
    sources = dict(c.fetchall())
    
    if not sources:
        st.warning("No data sources available. Please upload data first.")
        return
    
    # Select source
    selected_source = st.selectbox("Select Data Source", list(sources.values()))
    source_id = [k for k, v in sources.items() if v == selected_source][0]
    
    # Load data
    df = pd.read_sql(f"SELECT * FROM source_{source_id}", conn)
    
    # Analysis options
    analysis_type = st.selectbox("Select Analysis Type",
                               ["Descriptive Statistics", "Time Series Analysis", 
                                "Correlation Analysis", "Custom Visualization"])
    
    if analysis_type == "Descriptive Statistics":
        st.subheader("Statistical Summary")
        
        # Numeric statistics
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if not numeric_cols.empty:
            st.write("Numeric Columns Statistics:")
            st.dataframe(df[numeric_cols].describe())
        
        # Categorical statistics
        cat_cols = df.select_dtypes(include=['object']).columns
        if not cat_cols.empty:
            st.write("Categorical Columns Summary:")
            for col in cat_cols:
                st.write(f"\nValue Counts for {col}:")
                st.write(df[col].value_counts())
    
    elif analysis_type == "Time Series Analysis":
        date_cols = df.select_dtypes(include=['datetime64']).columns
        if date_cols.empty:
            st.warning("No datetime columns found. Please ensure your data includes properly formatted dates.")
            return
        
        date_col = st.selectbox("Select Date Column", date_cols)
        metric_col = st.selectbox("Select Metric to Analyze", 
                                df.select_dtypes(include=[np.number]).columns)
        
        # Resample frequency
        freq = st.selectbox("Select Time Frequency", 
                          ["Day", "Week", "Month", "Quarter", "Year

    elif analysis_type == "Time Series Analysis":
        date_cols = df.select_dtypes(include=['datetime64']).columns
        if date_cols.empty:
            st.warning("No datetime columns found. Please ensure your data includes properly formatted dates.")
            return
        
        date_col = st.selectbox("Select Date Column", date_cols)
        metric_col = st.selectbox("Select Metric to Analyze", 
                                df.select_dtypes(include=[np.number]).columns)
        
        # Resample frequency
        freq_map = {
            "Day": "D",
            "Week": "W",
            "Month": "M",
            "Quarter": "Q",
            "Year": "Y"
        }
        freq = st.selectbox("Select Time Frequency", list(freq_map.keys()))
        
        # Create time series plot
        df_ts = df.set_index(date_col)
        df_resampled = df_ts[metric_col].resample(freq_map[freq]).mean()
        
        fig = px.line(df_resampled, 
                     title=f'{metric_col} Over Time ({freq})',
                     labels={'value': metric_col, 'index': date_col})
        st.plotly_chart(fig)
        
        # Show trending metrics
        st.subheader("Trending Metrics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Trend Direction", 
                     "⬆️ Upward" if df_resampled.iloc[-1] > df_resampled.iloc[0] else "⬇️ Downward")
        with col2:
            st.metric("Total Change", 
                     f"{((df_resampled.iloc[-1] - df_resampled.iloc[0]) / df_resampled.iloc[0] * 100):.2f}%")
        with col3:
            st.metric("Volatility", 
                     f"{df_resampled.std():.2f}")
    
    elif analysis_type == "Correlation Analysis":
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 2:
            st.warning("Need at least 2 numeric columns for correlation analysis.")
            return
        
        st.subheader("Correlation Matrix")
        corr_matrix = df[numeric_cols].corr()
        
        # Heatmap
        fig = px.imshow(corr_matrix,
                       labels=dict(color="Correlation Coefficient"),
                       color_continuous_scale="RdBu")
        st.plotly_chart(fig)
        
        # Detailed correlation analysis
        st.subheader("Detailed Correlation Analysis")
        col1, col2 = st.columns(2)
        with col1:
            x_col = st.selectbox("Select X-axis Variable", numeric_cols)
        with col2:
            y_col = st.selectbox("Select Y-axis Variable", 
                               [col for col in numeric_cols if col != x_col])
        
        # Scatter plot
        fig = px.scatter(df, x=x_col, y=y_col, 
                        trendline="ols",
                        title=f'Scatter Plot: {x_col} vs {y_col}')
        st.plotly_chart(fig)
        
        # Correlation statistics
        correlation = df[x_col].corr(df[y_col])
        st.write(f"Correlation coefficient: {correlation:.4f}")
    
    elif analysis_type == "Custom Visualization":
        st.subheader("Create Custom Visualization")
        
        # Select chart type
        chart_type = st.selectbox("Select Chart Type", 
                                ["Bar Chart", "Scatter Plot", "Box Plot", "Histogram"])
        
        if chart_type == "Bar Chart":
            x_col = st.selectbox("Select X-axis Column", df.columns)
            y_col = st.selectbox("Select Y-axis Column", 
                               df.select_dtypes(include=[np.number]).columns)
            
            fig = px.bar(df, x=x_col, y=y_col,
                        title=f'Bar Chart: {y_col} by {x_col}')
            st.plotly_chart(fig)
        
        elif chart_type == "Scatter Plot":
            x_col = st.selectbox("Select X-axis Column", 
                               df.select_dtypes(include=[np.number]).columns)
            y_col = st.selectbox("Select Y-axis Column", 
                               [col for col in df.select_dtypes(include=[np.number]).columns 
                                if col != x_col])
            color_col = st.selectbox("Select Color Column (optional)", 
                                   ["None"] + df.columns.tolist())
            
            if color_col == "None":
                fig = px.scatter(df, x=x_col, y=y_col)
            else:
                fig = px.scatter(df, x=x_col, y=y_col, color=color_col)
            st.plotly_chart(fig)
        
        elif chart_type == "Box Plot":
            x_col = st.selectbox("Select Category Column", df.columns)
            y_col = st.selectbox("Select Value Column", 
                               df.select_dtypes(include=[np.number]).columns)
            
            fig = px.box(df, x=x_col, y=y_col,
                        title=f'Box Plot: {y_col} by {x_col}')
            st.plotly_chart(fig)
        
        elif chart_type == "Histogram":
            col = st.selectbox("Select Column", df.select_dtypes(include=[np.number]).columns)
            bins = st.slider("Number of Bins", 5, 100, 30)
            
            fig = px.histogram(df, x=col, nbins=bins,
                             title=f'Histogram: {col}')
            st.plotly_chart(fig)

# Data export function
def export_data():
    st.header("Data Export")
    
    conn = init_db()
    c = conn.cursor()
    
    # Get available data sources
    c.execute("SELECT id, name FROM data_sources")
    sources = dict(c.fetchall())
    
    if not sources:
        st.warning("No data sources available. Please upload data first.")
        return
    
    # Select source
    selected_source = st.selectbox("Select Data Source to Export", list(sources.values()))
    source_id = [k for k, v in sources.items() if v == selected_source][0]
    
    # Load data
    df = pd.read_sql(f"SELECT * FROM source_{source_id}", conn)
    
    # Export options
    export_format = st.selectbox("Select Export Format", ["CSV", "Excel", "JSON"])
    
    if st.button("Export Data"):
        try:
            if export_format == "CSV":
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            elif export_format == "Excel":
                buffer = io.BytesIO()
                df.to_excel(buffer, index=False)
                buffer.seek(0)
                st.download_button(
                    label="Download Excel",
                    data=buffer,
                    file_name=f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            elif export_format == "JSON":
                json_str = df.to_json(orient="records")
                st.download_button(
                    label="Download JSON",
                    data=json_str,
                    file_name=f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
            
            st.success("Data exported successfully!")
        except Exception as e:
            st.error(f"Error exporting data: {str(e)}")

# Main application
def main():
    st.title("🏭 Data Warehouse Management System")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", 
                           ["Data Ingestion",
                            "Data Transformation",
                            "Data Quality",
                            "Analytics & Reporting",
                            "Data Export"])
    
    # Page routing
    if page == "Data Ingestion":
        upload_data()
    elif page == "Data Transformation":
        transform_data()
    elif page == "Data Quality":
        manage_data_quality()
    elif page == "Analytics & Reporting":
        analyze_data()
    elif page == "Data Export":
        export_data()

if __name__ == "__main__":
    main()

    elif analysis_type == "Time Series Analysis":
        date_cols = df.select_dtypes(include=['datetime64']).columns
        if date_cols.empty:
            st.warning("No datetime columns found. Please ensure your data includes properly formatted dates.")
            return
        
        date_col = st.selectbox("Select Date Column", date_cols)
        metric_col = st.selectbox("Select Metric to Analyze", 
                                df.select_dtypes(include=[np.number]).columns)
        
        # Resample frequency
        freq_map = {
            "Day": "D",
            "Week": "W",
            "Month": "M",
            "Quarter": "Q",
            "Year": "Y"
        }
        freq = st.selectbox("Select Time Frequency", list(freq_map.keys()))
        
        # Create time series plot
        df_ts = df.set_index(date_col)
        df_resampled = df_ts[metric_col].resample(freq_map[freq]).mean()
        
        fig = px.line(df_resampled, 
                     title=f'{metric_col} Over Time ({freq})',
                     labels={'value': metric_col, 'index': date_col})
        st.plotly_chart(fig)
        
        # Show trending metrics
        st.subheader("Trending Metrics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Trend Direction", 
                     "⬆️ Upward" if df_resampled.iloc[-1] > df_resampled.iloc[0] else "⬇️ Downward")
        with col2:
            st.metric("Total Change", 
                     f"{((df_resampled.iloc[-1] - df_resampled.iloc[0]) / df_resampled.iloc[0] * 100):.2f}%")
        with col3:
            st.metric("Volatility", 
                     f"{df_resampled.std():.2f}")
    
    elif analysis_type == "Correlation Analysis":
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 2:
            st.warning("Need at least 2 numeric columns for correlation analysis.")
            return
        
        st.subheader("Correlation Matrix")
        corr_matrix = df[numeric_cols].corr()
        
        # Heatmap
        fig = px.imshow(corr_matrix,
                       labels=dict(color="Correlation Coefficient"),
                       color_continuous_scale="RdBu")
        st.plotly_chart(fig)
        
        # Detailed correlation analysis
        st.subheader("Detailed Correlation Analysis")
        col1, col2 = st.columns(2)
        with col1:
            x_col = st.selectbox("Select X-axis Variable", numeric_cols)
        with col2:
            y_col = st.selectbox("Select Y-axis Variable", 
                               [col for col in numeric_cols if col != x_col])
        
        # Scatter plot
        fig = px.scatter(df, x=x_col, y=y_col, 
                        trendline="ols",
                        title=f'Scatter Plot: {x_col} vs {y_col}')
        st.plotly_chart(fig)
        
        # Correlation statistics
        correlation = df[x_col].corr(df[y_col])
        st.write(f"Correlation coefficient: {correlation:.4f}")
    
    elif analysis_type == "Custom Visualization":
        st.subheader("Create Custom Visualization")
        
        # Select chart type
        chart_type = st.selectbox("Select Chart Type", 
                                ["Bar Chart", "Scatter Plot", "Box Plot", "Histogram"])
        
        if chart_type == "Bar Chart":
            x_col = st.selectbox("Select X-axis Column", df.columns)
            y_col = st.selectbox("Select Y-axis Column", 
                               df.select_dtypes(include=[np.number]).columns)
            
            fig = px.bar(df, x=x_col, y=y_col,
                        title=f'Bar Chart: {y_col} by {x_col}')
            st.plotly_chart(fig)
        
        elif chart_type == "Scatter Plot":
            x_col = st.selectbox("Select X-axis Column", 
                               df.select_dtypes(include=[np.number]).columns)
            y_col = st.selectbox("Select Y-axis Column", 
                               [col for col in df.select_dtypes(include=[np.number]).columns 
                                if col != x_col])
            color_col = st.selectbox("Select Color Column (optional)", 
                                   ["None"] + df.columns.tolist())
            
            if color_col == "None":
                fig = px.scatter(df, x=x_col, y=y_col)
            else:
                fig = px.scatter(df, x=x_col, y=y_col, color=color_col)
            st.plotly_chart(fig)
        
        elif chart_type == "Box Plot":
            x_col = st.selectbox("Select Category Column", df.columns)
            y_col = st.selectbox("Select Value Column", 
                               df.select_dtypes(include=[np.number]).columns)
            
            fig = px.box(df, x=x_col, y=y_col,
                        title=f'Box Plot: {y_col} by {x_col}')
            st.plotly_chart(fig)
        
        elif chart_type == "Histogram":
            col = st.selectbox("Select Column", df.select_dtypes(include=[np.number]).columns)
            bins = st.slider("Number of Bins", 5, 100, 30)
            
            fig = px.histogram(df, x=col, nbins=bins,
                             title=f'Histogram: {col}')
            st.plotly_chart(fig)

# Data export function
def export_data():
    st.header("Data Export")
    
    conn = init_db()
    c = conn.cursor()
    
    # Get available data sources
    c.execute("SELECT id, name FROM data_sources")
    sources = dict(c.fetchall())
    
    if not sources:
        st.warning("No data sources available. Please upload data first.")
        return
    
    # Select source
    selected_source = st.selectbox("Select Data Source to Export", list(sources.values()))
    source_id = [k for k, v in sources.items() if v == selected_source][0]
    
    # Load data
    df = pd.read_sql(f"SELECT * FROM source_{source_id}", conn)
    
    # Export options
    export_format = st.selectbox("Select Export Format", ["CSV", "Excel", "JSON"])
    
    if st.button("Export Data"):
        try:
            if export_format == "CSV":
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            elif export_format == "Excel":
                buffer = io.BytesIO()
                df.to_excel(buffer, index=False)
                buffer.seek(0)
                st.download_button(
                    label="Download Excel",
                    data=buffer,
                    file_name=f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            elif export_format == "JSON":
                json_str = df.to_json(orient="records")
                st.download_button(
                    label="Download JSON",
                    data=json_str,
                    file_name=f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
            
            st.success("Data exported successfully!")
        except Exception as e:
            st.error(f"Error exporting data: {str(e)}")

# Main application
def main():
    st.title("🏭 Data Warehouse Management System")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", 
                           ["Data Ingestion",
                            "Data Transformation",
                            "Data Quality",
                            "Analytics & Reporting",
                            "Data Export"])
    
    # Page routing
    if page == "Data Ingestion":
        upload_data()
    elif page == "Data Transformation":
        transform_data()
    elif page == "Data Quality":
        manage_data_quality()
    elif page == "Analytics & Reporting":
        analyze_data()
    elif page == "Data Export":
        export_data()

if __name__ == "__main__":
    main()
