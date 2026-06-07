import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="🌍 Country Intelligence System - Test",
    page_icon="🌍",
    layout="wide"
)

st.title("🌍 Country Intelligence System - Test Version")
st.markdown("## Testing Basic Functionality")

# Test data loading
st.markdown("### Upload Your Dataset")

uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
use_sample = st.checkbox("Or use default dataset (data/Country-data.csv)")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success(f"✅ Data loaded from upload! {df.shape[0]} rows, {df.shape[1]} columns")
    st.dataframe(df.head())
    
    # Basic statistics
    st.markdown("### Basic Statistics")
    st.write("**Shape:**", df.shape)
    st.write("**Columns:**", df.columns.tolist())
    st.write("**Missing values:**", df.isnull().sum().sum())

elif use_sample:
    try:
        df = pd.read_csv('data/Country-data.csv')
        st.success(f"✅ Data loaded from file! {df.shape[0]} rows, {df.shape[1]} columns")
        st.dataframe(df.head())
        
        # Basic statistics
        st.markdown("### Basic Statistics")
        st.write("**Shape:**", df.shape)
        st.write("**Columns:**", df.columns.tolist())
        st.write("**Missing values:**", df.isnull().sum().sum())
        
    except FileNotFoundError:
        st.error("❌ Could not find data/Country-data.csv file")
        st.info("Please make sure the data folder and CSV file exist")
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")

st.markdown("---")
st.success("✅ If you can see this page, Streamlit is working perfectly!")

# Test some basic functionality
st.markdown("### Basic Functionality Test")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Test Metric 1", "✅ Working")

with col2:
    st.metric("Test Metric 2", "✅ Working")

with col3:
    st.metric("Test Metric 3", "✅ Working")

# Test sidebar
with st.sidebar:
    st.title("Test Sidebar")
    st.success("Sidebar working!")
    test_slider = st.slider("Test Slider", 1, 10, 5)
    st.write(f"Slider value: {test_slider}")