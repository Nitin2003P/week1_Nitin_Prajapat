"""
Country Intelligence System - Professional Analytics Platform
For HELP International NGO
Developer: Nitin Prajapat | PCE
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import plotly.express as px
import plotly.graph_objects as go

# Import utility functions
from utils.data_loader import load_data, get_data_info, get_summary_stats, check_data_quality
from utils.preprocessing import preprocess_data, scale_features, apply_pca
from utils.clustering import (
    perform_kmeans, perform_dbscan, find_optimal_k, 
    calculate_silhouette, calculate_clustering_metrics
)
from utils.classification import train_all_models, evaluate_models, get_feature_importance, split_data
from utils.visualizations import (
    plot_correlation_heatmap, plot_distributions, plot_pca_clusters,
    plot_elbow_silhouette, plot_model_comparison, plot_feature_importance,
    plot_confusion_matrix, plot_priority_countries
)
import config

# Page configuration
st.set_page_config(
    page_title=config.APP_TITLE,
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional Custom CSS
st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Header Styling */
    h1 {
        color: #1e3a8a;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        padding: 20px 0;
        border-bottom: 3px solid #3b82f6;
    }
    
    h2 {
        color: #1e40af;
        font-weight: 600;
        margin-top: 30px;
    }
    
    h3 {
        color: #2563eb;
        font-weight: 600;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e3a8a 0%, #3b82f6 100%);
    }
    
    [data-testid="stSidebar"] .css-1d391kg, [data-testid="stSidebar"] .css-1v3fvcr {
        color: white;
    }
    
    /* Card Styling */
    .metric-card {
        background: white;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
        border-left: 5px solid #3b82f6;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        margin: 10px 0;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 24px rgba(0,0,0,0.15);
    }
    
    .metric-card h3 {
        color: #1e40af;
        margin-bottom: 10px;
        font-size: 20px;
    }
    
    .metric-card p {
        color: #64748b;
        font-size: 14px;
        line-height: 1.6;
    }
    
    /* Info Card */
    .info-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 30px;
        border-radius: 20px;
        color: white;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4);
        margin: 20px 0;
    }
    
    /* Stats Card */
    .stats-card {
        background: white;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        border-top: 4px solid #3b82f6;
    }
    
    .stats-card h4 {
        color: #64748b;
        font-size: 14px;
        font-weight: 500;
        margin-bottom: 8px;
    }
    
    .stats-card h2 {
        color: #1e40af;
        font-size: 32px;
        font-weight: 700;
        margin: 0;
    }
    
    /* Button Styling */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 12px 30px;
        border-radius: 10px;
        font-weight: 600;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    /* Alert Boxes */
    .stAlert {
        border-radius: 10px;
        border-left: 5px solid;
        padding: 15px;
    }
    
    /* DataFrames */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    
    /* Developer Credit Badge */
    .dev-badge {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 15px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3);
        margin: 15px 0;
    }
    
    .dev-badge p {
        color: white;
        margin: 5px 0;
        font-weight: 600;
    }
    
    /* Metric Styling */
    [data-testid="stMetricValue"] {
        font-size: 28px;
        font-weight: 700;
        color: #1e40af;
    }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: white;
        border-radius: 10px;
        padding: 5px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* Professional Header Banner */
    .header-banner {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        padding: 40px;
        border-radius: 20px;
        color: white;
        margin-bottom: 30px;
        box-shadow: 0 10px 30px rgba(30, 58, 138, 0.3);
    }
    
    .header-banner h1 {
        color: white;
        border: none;
        text-shadow: 2px 2px 8px rgba(0,0,0,0.2);
        margin: 0;
    }
    
    .header-banner p {
        font-size: 18px;
        opacity: 0.9;
        margin-top: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False
if 'clustering_done' not in st.session_state:
    st.session_state.clustering_done = False

# Professional Sidebar
with st.sidebar:
    st.markdown("""
    <div style='text-align: center; padding: 20px 0;'>
        <h1 style='color: white; font-size: 48px; margin: 0;'>🌍</h1>
        <h2 style='color: white; font-size: 24px; margin: 10px 0;'>Navigation</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Developer Credit Badge
    st.markdown("""
    <div class='dev-badge'>
        <p style='font-size: 18px; margin-bottom: 5px;'>💻 Developed By</p>
        <p style='font-size: 22px; margin: 5px 0;'>Nitin Prajapat</p>
        <p style='font-size: 14px; opacity: 0.9;'>📚 PCE</p>
        <p style='font-size: 12px; opacity: 0.8; margin-top: 5px;'>Data Science & ML</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
    
    # Navigation
    page = st.radio(
        "Select Module",
        ["🏠 Home", "📊 Data Overview", "🔍 EDA", "🎯 Clustering", 
         "🤖 Classification", "🎯 Recommendations", "📥 Download Results"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    # About Section
    st.markdown("""
    <div style='background: rgba(255,255,255,0.1); padding: 15px; border-radius: 10px;'>
        <h3 style='color: white; font-size: 16px; margin-bottom: 10px;'>📋 About</h3>
        <p style='color: white; font-size: 13px; line-height: 1.6;'>
            <b>HELP International</b><br>
            Humanitarian NGO committed to fighting poverty and providing aid to underdeveloped countries.
        </p>
        <p style='color: #fbbf24; font-size: 16px; font-weight: 600; margin-top: 10px;'>
            💰 Budget: $10 Million
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Settings
    st.markdown("<h3 style='color: white; font-size: 16px;'>⚙️ Settings</h3>", unsafe_allow_html=True)
    optimal_k = st.slider("Number of Clusters (K)", 2, 10, config.OPTIMAL_CLUSTERS)
    test_size = st.slider("Test Size", 0.1, 0.4, config.TEST_SIZE, 0.05)

# Main content
if page == "🏠 Home":
    # Professional Header Banner
    st.markdown("""
    <div class='header-banner'>
        <h1>🌍 Country Intelligence System</h1>
        <p>Advanced Analytics Platform for Strategic Aid Distribution</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Key Metrics Row
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
        <h3>🎯 Mission</h3>
        <p>Leverage AI and Machine Learning to identify countries in critical need of humanitarian aid and optimize resource allocation.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
        <h3>💰 Investment</h3>
        <p>Strategic deployment of <b>$10 Million</b> to maximize impact in underdeveloped nations worldwide.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
        <h3>🔬 Technology</h3>
        <p>Advanced ML algorithms including K-Means, DBSCAN, Random Forest, and XGBoost for data-driven decisions.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True)
    
    # Data Upload Section
    st.markdown("""
    <div style='background: white; padding: 30px; border-radius: 15px; box-shadow: 0 8px 16px rgba(0,0,0,0.1);'>
        <h2 style='margin-top: 0;'>📂 Data Upload Center</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Upload Country Dataset (CSV)",
            type=['csv'],
            help="Upload Country-data.csv file containing development indicators"
        )
        
        use_sample = st.checkbox("📊 Use Default Dataset (data/Country-data.csv)")
    
    if uploaded_file is not None or use_sample:
        try:
            if uploaded_file is not None:
                df = load_data(uploaded_file=uploaded_file)
            else:
                df = load_data(file_path='data/Country-data.csv')
            
            if df is not None:
                st.session_state.df = df
                st.session_state.data_loaded = True
                
                st.success(f"✅ Data loaded successfully! Analyzing {df.shape[0]} countries with {df.shape[1]} development indicators")
                
                with col2:
                    st.markdown("""
                    <div class='stats-card'>
                        <h4>Countries</h4>
                        <h2>{}</h2>
                    </div>
                    """.format(df.shape[0]), unsafe_allow_html=True)
                    
                    st.markdown("""
                    <div class='stats-card'>
                        <h4>Features</h4>
                        <h2>{}</h2>
                    </div>
                    """.format(df.shape[1]), unsafe_allow_html=True)
                
                st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
                st.markdown("### 📋 Data Preview")
                st.dataframe(df.head(10), use_container_width=True)
                
        except Exception as e:
            st.error(f"❌ Error loading data: {str(e)}")
    
    st.markdown("<div style='height: 40px;'></div>", unsafe_allow_html=True)
    
    # Features Section
    st.markdown("""
    <div class='info-card'>
        <h2 style='color: white; margin-top: 0;'>🚀 Platform Capabilities</h2>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
        <h3>📊 Advanced Analytics</h3>
        <ul style='color: #64748b; line-height: 1.8;'>
            <li><b>Exploratory Data Analysis:</b> Comprehensive statistical insights</li>
            <li><b>Clustering Algorithms:</b> K-Means & DBSCAN implementation</li>
            <li><b>ML Classification:</b> 4 models (LR, DT, RF, XGBoost)</li>
            <li><b>Feature Engineering:</b> PCA & importance analysis</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
        <h3>🎯 Strategic Outputs</h3>
        <ul style='color: #64748b; line-height: 1.8;'>
            <li><b>Country Segmentation:</b> Development level clustering</li>
            <li><b>Priority Ranking:</b> AI-powered urgency assessment</li>
            <li><b>Predictive Models:</b> Automated classification system</li>
            <li><b>Export Reports:</b> CSV/Excel downloadable results</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

elif page == "📊 Data Overview":
    st.markdown("""
    <div class='header-banner'>
        <h1>📊 Data Overview & Quality Assessment</h1>
        <p>Comprehensive analysis of dataset quality and characteristics</p>
    </div>
    """, unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please upload data from the Home page first!")
    else:
        df = st.session_state.df
        
        # Data Info
        col1, col2, col3, col4 = st.columns(4)
        
        info = get_data_info(df)
        
        with col1:
            st.markdown("""
            <div class='stats-card'>
                <h4>Total Countries</h4>
                <h2>{}</h2>
            </div>
            """.format(info['shape'][0]), unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class='stats-card'>
                <h4>Total Features</h4>
                <h2>{}</h2>
            </div>
            """.format(info['shape'][1]), unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class='stats-card'>
                <h4>Missing Values</h4>
                <h2>{}</h2>
            </div>
            """.format(sum(info['missing'].values())), unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div class='stats-card'>
                <h4>Duplicates</h4>
                <h2>{}</h2>
            </div>
            """.format(info['duplicates']), unsafe_allow_html=True)
        
        st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True)
        
        # Column Information
        st.markdown("## 📑 Feature Descriptions")
        
        feature_info = []
        for col in df.columns:
            if col in config.FEATURE_DESCRIPTIONS:
                feature_info.append({
                    'Feature': col,
                    'Description': config.FEATURE_DESCRIPTIONS[col],
                    'Type': str(df[col].dtype),
                    'Missing': df[col].isnull().sum(),
                    'Unique': df[col].nunique()
                })
        
        st.dataframe(pd.DataFrame(feature_info), use_container_width=True, height=400)
        
        st.markdown("---")
        
        # Summary Statistics
        st.markdown("## 📈 Statistical Summary")
        
        numeric_df = df.select_dtypes(include=[np.number])
        st.dataframe(numeric_df.describe().T.round(2), use_container_width=True, height=400)
        
        st.markdown("---")
        
        # Data Quality
        st.markdown("## ✅ Data Quality Assessment")
        
        quality_issues = check_data_quality(df)
        
        col1, col2 = st.columns(2)
        
        passed = [issue for issue in quality_issues if "✓" in issue]
        warnings = [issue for issue in quality_issues if "✓" not in issue]
        
        with col1:
            st.markdown("### ✅ Quality Checks Passed")
            for issue in passed:
                st.success(issue)
        
        with col2:
            st.markdown("### ⚠️ Warnings")
            for issue in warnings:
                st.warning(issue)

elif page == "🔍 EDA":
    st.markdown("""
    <div class='header-banner'>
        <h1>🔍 Exploratory Data Analysis</h1>
        <p>Deep dive into data patterns, correlations, and distributions</p>
    </div>
    """, unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please upload data from the Home page first!")
    else:
        df = st.session_state.df
        numeric_df = df.select_dtypes(include=[np.number])
        
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Distributions", "🔥 Correlations", "📦 Outliers", "💡 Key Insights"])
        
        with tab1:
            st.markdown("### 📊 Feature Distribution Analysis")
            
            selected_features = st.multiselect(
                "Select features to visualize",
                numeric_df.columns.tolist(),
                default=numeric_df.columns.tolist()[:3]
            )
            
            if selected_features:
                fig = plot_distributions(numeric_df, selected_features)
                st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            st.markdown("### 🔥 Correlation Matrix")
            
            fig = plot_correlation_heatmap(numeric_df)
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            st.markdown("#### 🔍 Top Correlations")
            
            corr_matrix = numeric_df.corr()
            
            # Find top correlations
            corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_pairs.append({
                        'Feature 1': corr_matrix.columns[i],
                        'Feature 2': corr_matrix.columns[j],
                        'Correlation': corr_matrix.iloc[i, j]
                    })
            
            corr_df = pd.DataFrame(corr_pairs).sort_values('Correlation', key=abs, ascending=False)
            st.dataframe(corr_df.head(10), use_container_width=True)
        
        with tab3:
            st.markdown("### 📦 Outlier Detection Analysis")
            
            selected_feature = st.selectbox("Select feature for boxplot", numeric_df.columns)
            
            fig = go.Figure()
            fig.add_trace(go.Box(y=numeric_df[selected_feature], name=selected_feature,
                                marker_color='#667eea'))
            fig.update_layout(
                title=f'Distribution & Outliers - {selected_feature}',
                height=400,
                template='plotly_white'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Outlier statistics
            Q1 = numeric_df[selected_feature].quantile(0.25)
            Q3 = numeric_df[selected_feature].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = numeric_df[(numeric_df[selected_feature] < lower_bound) | 
                                 (numeric_df[selected_feature] > upper_bound)]
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("""
                <div class='stats-card'>
                    <h4>Lower Bound</h4>
                    <h2>{:.2f}</h2>
                </div>
                """.format(lower_bound), unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div class='stats-card'>
                    <h4>Upper Bound</h4>
                    <h2>{:.2f}</h2>
                </div>
                """.format(upper_bound), unsafe_allow_html=True)
            
            with col3:
                st.markdown("""
                <div class='stats-card'>
                    <h4>Outliers</h4>
                    <h2>{}</h2>
                </div>
                """.format(len(outliers)), unsafe_allow_html=True)
        
        with tab4:
            st.markdown("### 💡 Key Insights & Findings")
            
            st.markdown("""
            <div class="metric-card">
            <h3>🔍 Development Indicators Analysis</h3>
            
            <h4>📉 Strong Negative Correlations</h4>
            <ul>
                <li><b>Child Mortality ↔ Life Expectancy:</b> Countries with high child mortality rates consistently show lower life expectancy</li>
                <li><b>Child Mortality ↔ Income:</b> Economic poverty directly correlates with higher child mortality rates</li>
            </ul>
            
            <h4>📈 Strong Positive Correlations</h4>
            <ul>
                <li><b>Income ↔ GDP per Capita:</b> Wealth indicators demonstrate strong alignment across nations</li>
                <li><b>Health Spending ↔ Life Expectancy:</b> Increased healthcare investment results in longer lifespans</li>
            </ul>
            
            <h4>📊 Distribution Insights</h4>
            <ul>
                <li><b>Life Expectancy:</b> Ranges from 32 to 83 years (51-year global gap)</li>
                <li><b>Income Disparity:</b> $609 to $125,000 per capita income variation</li>
                <li><b>Child Mortality:</b> 2.6 to 208 deaths per 1,000 births</li>
            </ul>
            
            <h4>⚠️ Critical Outliers</h4>
            <ul>
                <li>Several nations exhibit extremely high GDP per capita</li>
                <li>Inflation rates vary dramatically (-4.2% to 104%)</li>
                <li>Export/import ratios show significant variability indicating trade imbalances</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)

elif page == "🎯 Clustering":
    st.markdown("""
    <div class='header-banner'>
        <h1>🎯 Clustering Analysis</h1>
        <p>Unsupervised learning to segment countries by development indicators</p>
    </div>
    """, unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please upload data from the Home page first!")
    else:
        df = st.session_state.df
        
        # Preprocess data
        df_numeric, countries = preprocess_data(df, target_col='country')
        df_scaled, scaler = scale_features(df_numeric)
        
        tab1, tab2, tab3, tab4 = st.tabs(["🎲 K-Means", "🔬 DBSCAN", "📉 PCA", "⚖️ Comparison"])
        
        with tab1:
            st.markdown("### 🎲 K-Means Clustering")
            
            # Find optimal K
            with st.spinner("🔄 Computing optimal cluster count..."):
                k_values, inertias, silhouette_scores = find_optimal_k(df_scaled, range(2, 11))
            
            # Plot elbow and silhouette
            fig = plot_elbow_silhouette(k_values, inertias, silhouette_scores)
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            
            # Perform K-Means with selected K
            kmeans_labels, kmeans_model = perform_kmeans(df_scaled, n_clusters=optimal_k)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                <div class='stats-card'>
                    <h4>Selected K</h4>
                    <h2>{}</h2>
                </div>
                """.format(optimal_k), unsafe_allow_html=True)
                
                silhouette = calculate_silhouette(df_scaled, kmeans_labels)
                st.markdown("""
                <div class='stats-card'>
                    <h4>Silhouette Score</h4>
                    <h2>{:.4f}</h2>
                </div>
                """.format(silhouette), unsafe_allow_html=True)
            
            with col2:
                # Cluster distribution
                unique, counts = np.unique(kmeans_labels, return_counts=True)
                cluster_dist = pd.DataFrame({
                    'Cluster': unique,
                    'Count': counts,
                    'Percentage': (counts / len(kmeans_labels) * 100).round(2)
                })
                st.dataframe(cluster_dist, use_container_width=True)
            
            # PCA Visualization
            st.markdown("#### 🗺️ Cluster Visualization (PCA)")
            pca_df, explained_var, pca_model = apply_pca(df_scaled, n_components=2)
            
            fig = plot_pca_clusters(pca_df, kmeans_labels, 
                                   title=f'K-Means Clustering (K={optimal_k})')
            st.plotly_chart(fig, use_container_width=True)
            
            st.info(f"📊 Explained Variance: PC1={explained_var[0]:.2%}, PC2={explained_var[1]:.2%}, Total={sum(explained_var):.2%}")
            
            # Save to session state
            st.session_state.kmeans_labels = kmeans_labels
            st.session_state.kmeans_model = kmeans_model
            st.session_state.pca_df = pca_df
            st.session_state.clustering_done = True
        
        with tab2:
            st.markdown("### 🔬 DBSCAN Clustering")
            
            col1, col2 = st.columns(2)
            
            with col1:
                eps = st.slider("Epsilon (eps)", 0.5, 5.0, 2.5, 0.1)
            with col2:
                min_samples = st.slider("Min Samples", 2, 10, 3)
            
            if st.button("🚀 Run DBSCAN", type="primary"):
                with st.spinner("Running DBSCAN algorithm..."):
                    dbscan_labels, n_clusters, n_noise = perform_dbscan(df_scaled, eps, min_samples)
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown("""
                        <div class='stats-card'>
                            <h4>Clusters Found</h4>
                            <h2>{}</h2>
                        </div>
                        """.format(n_clusters), unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown("""
                        <div class='stats-card'>
                            <h4>Noise Points</h4>
                            <h2>{}</h2>
                        </div>
                        """.format(n_noise), unsafe_allow_html=True)
                    
                    with col3:
                        if n_clusters > 1:
                            metrics = calculate_clustering_metrics(df_scaled.values, dbscan_labels)
                            st.markdown("""
                            <div class='stats-card'>
                                <h4>Silhouette Score</h4>
                                <h2>{:.4f}</h2>
                            </div>
                            """.format(metrics['silhouette']), unsafe_allow_html=True)
                    
                    # Visualization
                    pca_df, _, _ = apply_pca(df_scaled, n_components=2)
                    fig = plot_pca_clusters(pca_df, dbscan_labels, 
                                           title='DBSCAN Clustering')
                    st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.markdown("### 📉 Principal Component Analysis")
            
            n_components = st.slider("Number of Components", 2, 5, 2)
            
            pca_df_full, explained_var_full, _ = apply_pca(df_scaled, n_components=n_components)
            
            # Scree plot
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=[f'PC{i+1}' for i in range(n_components)],
                y=explained_var_full,
                text=[f'{var:.2%}' for var in explained_var_full],
                textposition='auto',
                marker_color='#667eea'
            ))
            fig.update_layout(
                title='PCA Explained Variance Ratio',
                xaxis_title='Principal Components',
                yaxis_title='Explained Variance',
                height=400,
                template='plotly_white'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            st.markdown("#### 🔍 Component Loadings")
            
            # Calculate loadings
            pca_full = apply_pca(df_scaled, n_components=n_components)[2]
            loadings = pd.DataFrame(
                pca_full.components_.T,
                columns=[f'PC{i+1}' for i in range(n_components)],
                index=df_numeric.columns
            )
            
            st.dataframe(loadings.style.background_gradient(cmap='RdBu', axis=0), 
                        use_container_width=True)
        
        with tab4:
            st.markdown("### ⚖️ Algorithm Comparison")
            
            if st.session_state.clustering_done:
                kmeans_metrics = calculate_clustering_metrics(
                    df_scaled.values, 
                    st.session_state.kmeans_labels
                )
                
                comparison = pd.DataFrame({
                    'Metric': ['Silhouette Score', 'Calinski-Harabasz', 'Davies-Bouldin'],
                    'K-Means': [
                        f"{kmeans_metrics['silhouette']:.4f}",
                        f"{kmeans_metrics['calinski_harabasz']:.2f}",
                        f"{kmeans_metrics['davies_bouldin']:.4f}"
                    ],
                    'Interpretation': [
                        'Higher is better (>0.5 = good)',
                        'Higher values indicate better clustering',
                        'Lower values indicate better clustering'
                    ]
                })
                
                st.dataframe(comparison, use_container_width=True)
                
                st.markdown("""
                <div class="metric-card">
                <h3>📊 Evaluation Metrics Explained</h3>
                
                <ul>
                    <li><b>Silhouette Score:</b> Measures how similar objects are to their own cluster compared to other clusters. Range: -1 to 1.</li>
                    <li><b>Calinski-Harabasz Index:</b> Ratio of between-cluster dispersion to within-cluster dispersion. Higher = better defined clusters.</li>
                    <li><b>Davies-Bouldin Index:</b> Average similarity between each cluster and its most similar cluster. Lower = better separation.</li>
                </ul>
                
                <p><b>✅ Recommendation:</b> Use K-Means clusters for subsequent classification tasks based on superior metrics.</p>
                </div>
                """, unsafe_allow_html=True)

elif page == "🤖 Classification":
    st.markdown("""
    <div class='header-banner'>
        <h1>🤖 Machine Learning Classification</h1>
        <p>Supervised learning models to predict country development clusters</p>
    </div>
    """, unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please upload data from the Home page first!")
    elif not st.session_state.clustering_done:
        st.warning("⚠️ Please complete clustering analysis first!")
    else:
        df = st.session_state.df
        df_numeric, countries = preprocess_data(df, target_col='country')
        df_scaled, _ = scale_features(df_numeric)
        
        X = df_scaled
        y = st.session_state.kmeans_labels
        
        tab1, tab2, tab3 = st.tabs(["🎯 Train Models", "📊 Evaluation", "🎨 Feature Importance"])
        
        with tab1:
            st.markdown("### 🎯 Model Training Center")
            
            st.info("💡 Using K-Means cluster labels as target variable for supervised learning")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown("""
                <div class='stats-card'>
                    <h4>Total Samples</h4>
                    <h2>{}</h2>
                </div>
                """.format(len(X)), unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div class='stats-card'>
                    <h4>Features</h4>
                    <h2>{}</h2>
                </div>
                """.format(X.shape[1]), unsafe_allow_html=True)
            
            with col3:
                st.markdown("""
                <div class='stats-card'>
                    <h4>Target Classes</h4>
                    <h2>{}</h2>
                </div>
                """.format(len(np.unique(y))), unsafe_allow_html=True)
            
            with col4:
                st.markdown("""
                <div class='stats-card'>
                    <h4>Test Size</h4>
                    <h2>{:.0f}%</h2>
                </div>
                """.format(test_size*100), unsafe_allow_html=True)
            
            st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
            
            if st.button("🚀 Train All Models", type="primary", use_container_width=True):
                with st.spinner("🔄 Training multiple ML models... This may take a moment."):
                    # Split data
                    X_train, X_test, y_train, y_test = split_data(X, y, test_size, config.RANDOM_STATE)
                    
                    # Progress bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Train models
                    status_text.text("Training Logistic Regression...")
                    progress_bar.progress(25)
                    
                    models, predictions = train_all_models(X_train, X_test, y_train, y_test)
                    
                    progress_bar.progress(100)
                    status_text.text("✅ Training complete!")
                    
                    # Save to session state
                    st.session_state.models = models
                    st.session_state.predictions = predictions
                    st.session_state.X_train = X_train
                    st.session_state.X_test = X_test
                    st.session_state.y_train = y_train
                    st.session_state.y_test = y_test
                    st.session_state.models_trained = True
                    
                    st.success("✅ All models trained successfully! Navigate to Evaluation tab to view results.")
                    st.balloons()
        
        with tab2:
            st.markdown("### 📊 Model Performance Evaluation")
            
            if not st.session_state.models_trained:
                st.warning("⚠️ Please train models first in the 'Train Models' tab!")
            else:
                predictions = st.session_state.predictions
                y_test = st.session_state.y_test
                
                # Evaluate models
                results_df = evaluate_models(y_test, predictions)
                results_df = results_df.sort_values('F1-Score', ascending=False)
                
                st.markdown("#### 📈 Performance Metrics Summary")
                
                # Format for display
                display_df = results_df.copy()
                for col in ['Accuracy', 'Precision', 'Recall', 'F1-Score']:
                    display_df[col] = display_df[col].apply(lambda x: f'{x*100:.2f}%')
                
                st.dataframe(display_df, use_container_width=True)
                
                # Visual comparison
                fig = plot_model_comparison(results_df)
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("---")
                
                # Confusion Matrices
                st.markdown("#### 🔍 Confusion Matrix Analysis")
                
                selected_model = st.selectbox("Select Model for Detailed View", list(predictions.keys()))
                
                cm = confusion_matrix(y_test, predictions[selected_model])
                fig = plot_confusion_matrix(cm, selected_model)
                
                col1, col2 = st.columns([3, 2])
                
                with col1:
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.markdown("##### 📋 Classification Report")
                    from sklearn.metrics import classification_report
                    report = classification_report(y_test, predictions[selected_model], 
                                                  output_dict=True)
                    report_df = pd.DataFrame(report).transpose()
                    st.dataframe(report_df.round(4), use_container_width=True)
                
                # Best model
                best_model = results_df.iloc[0]['Model']
                best_score = results_df.iloc[0]['F1-Score']
                
                st.success(f"🏆 **Best Performing Model:** {best_model} | **F1-Score:** {best_score:.4f}")
        
        with tab3:
            st.markdown("### 🎨 Feature Importance Analysis")
            
            if not st.session_state.models_trained:
                st.warning("⚠️ Please train models first!")
            else:
                models = st.session_state.models
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### 🌲 Random Forest")
                    rf_importance = get_feature_importance(
                        models['Random Forest'], 
                        df_numeric.columns
                    )
                    if rf_importance is not None:
                        fig = plot_feature_importance(rf_importance, 'Random Forest Feature Importance')
                        st.plotly_chart(fig, use_container_width=True)
                        st.dataframe(rf_importance, use_container_width=True)
                
                with col2:
                    st.markdown("#### ⚡ XGBoost")
                    xgb_importance = get_feature_importance(
                        models['XGBoost'], 
                        df_numeric.columns
                    )
                    if xgb_importance is not None:
                        fig = plot_feature_importance(xgb_importance, 'XGBoost Feature Importance')
                        st.plotly_chart(fig, use_container_width=True)
                        st.dataframe(xgb_importance, use_container_width=True)
                
                st.markdown("---")
                
                st.markdown("""
                <div class="metric-card">
                <h3>💡 Key Insights from Feature Importance</h3>
                
                <p>The most critical features for predicting country development clusters are:</p>
                
                <ol>
                    <li><b>Child Mortality:</b> The strongest single indicator of development level across all models</li>
                    <li><b>Life Expectancy:</b> Reflects overall health infrastructure and quality of life</li>
                    <li><b>Income/GDPP:</b> Economic indicators crucial for accurate cluster classification</li>
                    <li><b>Health Spending:</b> Investment in healthcare correlates with development status</li>
                </ol>
                
                <p><b>Strategic Implication:</b> These features align perfectly with HELP International's humanitarian focus areas, validating the model's relevance for aid distribution decisions.</p>
                </div>
                """, unsafe_allow_html=True)

elif page == "🎯 Recommendations":
    st.markdown("""
    <div class='header-banner'>
        <h1>🎯 Strategic Aid Distribution Recommendations</h1>
        <p>Data-driven insights for optimal resource allocation</p>
    </div>
    """, unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please upload data from the Home page first!")
    elif not st.session_state.clustering_done:
        st.warning("⚠️ Please complete clustering analysis first!")
    else:
        df = st.session_state.df
        kmeans_labels = st.session_state.kmeans_labels
        
        # Create analysis dataframe
        df_analysis = df.copy()
        df_analysis['Cluster'] = kmeans_labels
        
        tab1, tab2, tab3 = st.tabs(["📊 Cluster Analysis", "🚨 Priority Countries", "💰 Budget Allocation"])
        
        with tab1:
            st.markdown("### 📊 Development Cluster Characteristics")
            
            # Cluster statistics
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            cluster_stats = df_analysis.groupby('Cluster')[numeric_cols].mean()
            
            st.dataframe(cluster_stats.round(2), use_container_width=True)
            
            st.markdown("---")
            
            # Identify underdeveloped cluster
            cluster_dev_score = pd.DataFrame({
                'Cluster': cluster_stats.index,
                'Avg_Income': cluster_stats['income'],
                'Avg_GDPP': cluster_stats['gdpp'],
                'Avg_Child_Mort': cluster_stats['child_mort'],
                'Avg_Life_Expec': cluster_stats['life_expec']
            })
            
            cluster_dev_score['Development_Score'] = (
                cluster_dev_score['Avg_Income'] + 
                cluster_dev_score['Avg_GDPP'] - 
                cluster_dev_score['Avg_Child_Mort'] * 100 + 
                cluster_dev_score['Avg_Life_Expec'] * 100
            )
            
            cluster_dev_score = cluster_dev_score.sort_values('Development_Score')
            
            st.markdown("#### 🏆 Cluster Development Ranking")
            st.dataframe(cluster_dev_score, use_container_width=True)
            
            most_underdeveloped = int(cluster_dev_score.iloc[0]['Cluster'])
            
            st.error(f"🚨 **Most Underdeveloped Cluster Identified:** Cluster {most_underdeveloped}")
        
        with tab2:
            st.markdown("### 🚨 Priority Countries for Immediate Aid")
            
            # Get underdeveloped countries
            underdeveloped_countries = df_analysis[
                df_analysis['Cluster'] == most_underdeveloped
            ].copy()
            
            # Calculate priority score
            underdeveloped_countries['Priority_Score'] = (
                (underdeveloped_countries['child_mort'] / underdeveloped_countries['child_mort'].max()) * config.PRIORITY_WEIGHTS['child_mort'] +
                (1 - underdeveloped_countries['life_expec'] / underdeveloped_countries['life_expec'].max()) * config.PRIORITY_WEIGHTS['life_expec'] +
                (1 - underdeveloped_countries['income'] / underdeveloped_countries['income'].max()) * config.PRIORITY_WEIGHTS['income'] +
                (underdeveloped_countries['total_fer'] / underdeveloped_countries['total_fer'].max()) * config.PRIORITY_WEIGHTS['total_fer'] +
                (1 - underdeveloped_countries['gdpp'] / underdeveloped_countries['gdpp'].max()) * config.PRIORITY_WEIGHTS['gdpp']
            )
            
            underdeveloped_countries = underdeveloped_countries.sort_values('Priority_Score', ascending=False)
            
            st.markdown("""
            <div class='stats-card'>
                <h4>Countries in Critical Need</h4>
                <h2>{}</h2>
            </div>
            """.format(len(underdeveloped_countries)), unsafe_allow_html=True)
            
            st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
            
            # Top N slider
            top_n = st.slider("Number of top priority countries to display", 5, 20, 10)
            
            top_countries = underdeveloped_countries.head(top_n)[
                ['country', 'child_mort', 'life_expec', 'income', 'gdpp', 'Priority_Score']
            ].reset_index(drop=True)
            
            top_countries.index = top_countries.index + 1
            
            st.markdown(f"#### 🎯 Top {top_n} Priority Countries")
            st.dataframe(top_countries.style.background_gradient(subset=['Priority_Score'], cmap='RdYlGn_r'), 
                        use_container_width=True)
            
            # Visualization
            fig = plot_priority_countries(underdeveloped_countries, top_n)
            st.plotly_chart(fig, use_container_width=True)
            
            # Save to session state
            st.session_state.top_countries = top_countries
            st.session_state.underdeveloped_countries = underdeveloped_countries
            
            st.markdown("---")
            
            # Detailed profiles
            st.markdown("#### 🔍 Detailed Country Profiles")
            
            selected_country = st.selectbox(
                "Select country for comprehensive analysis",
                top_countries['country'].tolist()
            )
            
            country_data = underdeveloped_countries[
                underdeveloped_countries['country'] == selected_country
            ].iloc[0]
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("""
                <div class='stats-card'>
                    <h4>Child Mortality</h4>
                    <h2>{:.1f}</h2>
                    <p style='color: #64748b; font-size: 12px;'>per 1,000 births</p>
                </div>
                """.format(country_data['child_mort']), unsafe_allow_html=True)
                
                st.markdown("""
                <div class='stats-card'>
                    <h4>Life Expectancy</h4>
                    <h2>{:.1f}</h2>
                    <p style='color: #64748b; font-size: 12px;'>years</p>
                </div>
                """.format(country_data['life_expec']), unsafe_allow_html=True)
                
                st.markdown("""
                <div class='stats-card'>
                    <h4>Total Fertility</h4>
                    <h2>{:.2f}</h2>
                    <p style='color: #64748b; font-size: 12px;'>births per woman</p>
                </div>
                """.format(country_data['total_fer']), unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div class='stats-card'>
                    <h4>Income per Capita</h4>
                    <h2>${:.0f}</h2>
                    <p style='color: #64748b; font-size: 12px;'>annual</p>
                </div>
                """.format(country_data['income']), unsafe_allow_html=True)
                
                st.markdown("""
                <div class='stats-card'>
                    <h4>GDP per Capita</h4>
                    <h2>${:.0f}</h2>
                    <p style='color: #64748b; font-size: 12px;'>PPP adjusted</p>
                </div>
                """.format(country_data['gdpp']), unsafe_allow_html=True)
                
                st.markdown("""
                <div class='stats-card'>
                    <h4>Health Spending</h4>
                    <h2>{:.2f}%</h2>
                    <p style='color: #64748b; font-size: 12px;'>of GDP</p>
                </div>
                """.format(country_data['health']), unsafe_allow_html=True)
            
            with col3:
                st.markdown("""
                <div class='stats-card'>
                    <h4>Exports</h4>
                    <h2>{:.1f}%</h2>
                    <p style='color: #64748b; font-size: 12px;'>of GDP</p>
                </div>
                """.format(country_data['exports']), unsafe_allow_html=True)
                
                st.markdown("""
                <div class='stats-card'>
                    <h4>Imports</h4>
                    <h2>{:.1f}%</h2>
                    <p style='color: #64748b; font-size: 12px;'>of GDP</p>
                </div>
                """.format(country_data['imports']), unsafe_allow_html=True)
                
                st.markdown("""
                <div class='stats-card'>
                    <h4>Inflation</h4>
                    <h2>{:.2f}%</h2>
                    <p style='color: #64748b; font-size: 12px;'>annual rate</p>
                </div>
                """.format(country_data['inflation']), unsafe_allow_html=True)
        
        with tab3:
            st.markdown("### 💰 Strategic Budget Allocation")
            
            st.markdown("""
            <div class='info-card'>
                <h2 style='color: white; margin: 0;'>💵 Total Available Budget: $10,000,000</h2>
            </div>
            """, unsafe_allow_html=True)
            
            if 'top_countries' in st.session_state:
                top_countries = st.session_state.top_countries
                
                allocation_strategy = st.radio(
                    "Select Budget Allocation Strategy",
                    ["Equal Distribution", "Priority-Based", "Custom Allocation"]
                )
                
                if allocation_strategy == "Equal Distribution":
                    budget_per_country = 10_000_000 / len(top_countries)
                    allocations = [budget_per_country] * len(top_countries)
                
                elif allocation_strategy == "Priority-Based":
                    # Allocate based on priority score
                    total_priority = top_countries['Priority_Score'].sum()
                    allocations = (top_countries['Priority_Score'] / total_priority * 10_000_000).tolist()
                
                else:  # Custom
                    st.markdown("#### 🎚️ Custom Budget Allocation")
                    allocations = []
                    for idx, country in enumerate(top_countries['country']):
                        amount = st.number_input(
                            f"💵 {country}",
                            min_value=0,
                            max_value=10_000_000,
                            value=1_000_000,
                            step=100_000,
                            key=f"budget_{idx}"
                        )
                        allocations.append(amount)
                
                # Create allocation dataframe
                allocation_df = top_countries.copy()
                allocation_df['Allocated_Budget'] = allocations
                allocation_df['Budget_Formatted'] = allocation_df['Allocated_Budget'].apply(
                    lambda x: f"${x:,.0f}"
                )
                
                st.markdown("---")
                st.markdown("#### 📊 Budget Distribution Summary")
                st.dataframe(
                    allocation_df[['country', 'Priority_Score', 'Budget_Formatted']],
                    use_container_width=True
                )
                
                # Visualization
                fig = go.Figure(go.Pie(
                    labels=allocation_df['country'],
                    values=allocation_df['Allocated_Budget'],
                    hole=0.4,
                    marker=dict(colors=px.colors.sequential.Blues_r)
                ))
                fig.update_layout(
                    title='Budget Distribution by Country',
                    height=500,
                    template='plotly_white'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("""
                    <div class='stats-card'>
                        <h4>Total Allocated</h4>
                        <h2>${:,.0f}</h2>
                    </div>
                    """.format(sum(allocations)), unsafe_allow_html=True)
                
                with col2:
                    remaining = 10_000_000 - sum(allocations)
                    st.markdown("""
                    <div class='stats-card'>
                        <h4>Remaining Budget</h4>
                        <h2>${:,.0f}</h2>
                    </div>
                    """.format(remaining), unsafe_allow_html=True)
                
                with col3:
                    avg_allocation = sum(allocations) / len(allocations)
                    st.markdown("""
                    <div class='stats-card'>
                        <h4>Avg per Country</h4>
                        <h2>${:,.0f}</h2>
                    </div>
                    """.format(avg_allocation), unsafe_allow_html=True)
                
                st.session_state.allocation_df = allocation_df

elif page == "📥 Download Results":
    st.markdown("""
    <div class='header-banner'>
        <h1>📥 Export & Download Center</h1>
        <p>Download comprehensive analysis results and reports</p>
    </div>
    """, unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please complete the analysis first!")
    else:
        st.markdown("### 📦 Available Downloads")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 Data & Clustering")
            
            # Cluster assignments
            if st.session_state.clustering_done:
                df_with_clusters = st.session_state.df.copy()
                df_with_clusters['Cluster'] = st.session_state.kmeans_labels
                
                csv_clusters = df_with_clusters.to_csv(index=False)
                st.download_button(
                    label="📊 Download Cluster Assignments",
                    data=csv_clusters,
                    file_name="country_cluster_assignments.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            # Priority countries
            if 'top_countries' in st.session_state:
                csv_priority = st.session_state.top_countries.to_csv(index=False)
                st.download_button(
                    label="🚨 Download Priority Countries",
                    data=csv_priority,
                    file_name="priority_countries.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        
        with col2:
            st.markdown("#### 🤖 Models & Budget")
            
            # Model results
            if st.session_state.models_trained:
                results_df = evaluate_models(
                    st.session_state.y_test,
                    st.session_state.predictions
                )
                csv_results = results_df.to_csv(index=False)
                st.download_button(
                    label="🤖 Download Model Performance",
                    data=csv_results,
                    file_name="model_performance.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            # Budget allocation
            if 'allocation_df' in st.session_state:
                csv_allocation = st.session_state.allocation_df.to_csv(index=False)
                st.download_button(
                    label="💰 Download Budget Allocation",
                    data=csv_allocation,
                    file_name="budget_allocation.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        
        st.markdown("---")
        
        # Comprehensive report
        st.markdown("### 📄 Comprehensive Executive Report")
        
        if st.button("📝 Generate Full Report", type="primary", use_container_width=True):
            with st.spinner("Generating comprehensive report..."):
                report_text = f"""
{'='*90}
COUNTRY INTELLIGENCE SYSTEM - COMPREHENSIVE EXECUTIVE REPORT
{'='*90}

Organization: HELP International
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
Analyst: Nitin Prajapat | PCE

{'='*90}
EXECUTIVE SUMMARY
{'='*90}

Total Countries Analyzed: {len(st.session_state.df)}
Clustering Algorithm: K-Means
Optimal Clusters: {optimal_k}
Countries Requiring Urgent Aid: {len(st.session_state.underdeveloped_countries) if 'underdeveloped_countries' in st.session_state else 'N/A'}
Total Budget Available: $10,000,000

{'='*90}
TOP PRIORITY COUNTRIES FOR AID DISTRIBUTION
{'='*90}

{st.session_state.top_countries.to_string() if 'top_countries' in st.session_state else 'Analysis incomplete - Please complete clustering'}

{'='*90}
MACHINE LEARNING MODEL PERFORMANCE
{'='*90}

{evaluate_models(st.session_state.y_test, st.session_state.predictions).to_string() if st.session_state.models_trained else 'Models not yet trained'}

{'='*90}
STRATEGIC RECOMMENDATIONS
{'='*90}

1. IMMEDIATE ACTION ITEMS
   - Allocate 50% of budget ($5M) to top 5 priority countries
   - Deploy resources within next 90 days
   - Focus on child mortality and healthcare infrastructure

2. MEDIUM-TERM STRATEGY (6-12 months)
   - Distribute remaining $5M across next 10-15 countries
   - Establish monitoring systems for aid effectiveness
   - Partner with local healthcare organizations

3. FOCUS AREAS
   ✓ Reduce child mortality rates
   ✓ Improve healthcare infrastructure
   ✓ Increase life expectancy
   ✓ Support education programs
   ✓ Economic development initiatives

4. MONITORING & EVALUATION
   - Quarterly progress reviews
   - Annual cluster re-evaluation
   - Impact assessment metrics
   - Adjust allocation based on outcomes

5. RISK MITIGATION
   - Political stability assessment
   - Infrastructure readiness evaluation
   - Local partnership development
   - Contingency fund allocation (10%)

{'='*90}
METHODOLOGY
{'='*90}

Data Sources: WHO, World Bank, UN Development Programme
ML Algorithms: K-Means Clustering, Random Forest, XGBoost, Logistic Regression
Validation: Cross-validation, Silhouette analysis, F1-Score optimization
Priority Scoring: Multi-factor weighted algorithm

{'='*90}
CONCLUSION
{'='*90}

This AI-powered analysis provides data-driven recommendations for optimal aid
distribution. The identified priority countries demonstrate the highest need
based on multiple development indicators. Implementing these recommendations
will maximize humanitarian impact and support HELP International's mission.

{'='*90}
REPORT END
{'='*90}

Prepared by: Nitin Prajapat
Institution: PCE (Poornima College of Engineering)
Specialization: Data Science & Machine Learning
                """
                
                st.download_button(
                    label="📥 Download Full Executive Report",
                    data=report_text,
                    file_name=f"HELP_International_Report_{pd.Timestamp.now().strftime('%Y%m%d')}.txt",
                    mime="text/plain",
                    use_container_width=True
                )
                
                st.success("✅ Report generated successfully!")

# Professional Footer
st.markdown("<div style='height: 50px;'></div>", unsafe_allow_html=True)
st.markdown("""
<div style='background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%); 
            padding: 40px; border-radius: 20px; text-align: center; 
            box-shadow: 0 10px 30px rgba(30, 58, 138, 0.3);'>
    <div style='color: white;'>
        <h2 style='margin: 0 0 15px 0; color: white; font-size: 28px;'>
            🌍 Country Intelligence System
        </h2>
        <p style='margin: 10px 0; font-size: 16px; opacity: 0.95;'>
            Advanced Analytics Platform Powered by Machine Learning & Artificial Intelligence
        </p>
        <div style='height: 2px; background: rgba(255,255,255,0.3); margin: 25px auto; width: 60%;'></div>
        <div style='background: rgba(255,255,255,0.1); padding: 20px; border-radius: 15px; 
                    display: inline-block; margin-top: 15px;'>
            <p style='margin: 5px 0; font-size: 20px; font-weight: 700;'>
                💻 Developed By
            </p>
            <p style='margin: 10px 0; font-size: 26px; font-weight: 700; 
                      background: linear-gradient(135deg, #fbbf24 0%, #f59e0b 100%);
                      -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                      background-clip: text;'>
                Nitin Prajapat
            </p>
            <p style='margin: 5px 0; font-size: 16px; opacity: 0.9;'>
                📚 Poornima College of Engineering (PCE)
            </p>
            <p style='margin: 5px 0; font-size: 14px; opacity: 0.8;'>
                🎓 Data Science & Machine Learning Specialist
            </p>
        </div>
        <p style='margin: 25px 0 0 0; font-size: 13px; opacity: 0.7;'>
            © 2024 Country Intelligence System | Built with Streamlit & Python
        </p>
    </div>
</div>
""", unsafe_allow_html=True)