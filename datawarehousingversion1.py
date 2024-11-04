# ... (previous code remains the same until the time series analysis section)

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
