# ===========================================
# 📊 Bank Customer Attrition Analysis App
# ===========================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# ===========================================
# 🔧 Page Configuration
# ===========================================
st.set_page_config(
    page_title="Bank Churn Analysis",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===========================================
# 🎨 Custom CSS Styling
# ===========================================
st.markdown("""
<style>
    /* Main headers */
    .main-title {
        font-size: 2.8rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
        padding-bottom: 1rem;
        border-bottom: 3px solid #2E8B57;
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.8rem;
        color: #4682B4;
        margin-top: 2rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #4682B4;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 6px solid #2E8B57;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
    }
    
    /* Success/Error messages */
    .stSuccess {
        border-radius: 10px;
        padding: 1rem;
    }
    
    .stWarning {
        border-radius: 10px;
        padding: 1rem;
    }
    
    /* Dataframe styling */
    .stDataFrame {
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #2E8B57 0%, #3CB371 100%);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 8px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 4px 12px rgba(46, 139, 87, 0.4);
    }
    
    /* Sidebar */
    .css-1d391kg {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

# ===========================================
# 🏦 Main Title
# ===========================================
st.markdown('<h1 class="main-title">🏦 Bank Customer Churn Analysis Dashboard</h1>', unsafe_allow_html=True)
st.markdown("#### 🔍 Comprehensive Analysis of Banking Customer Attrition Factors")

# ===========================================
# 📂 Sidebar Navigation
# ===========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2830/2830284.png", width=120)
    st.markdown("---")
    
    st.markdown("### 📋 Main Menu")
    page = st.radio(
        "Select Section:",
        ["📊 Data Overview", 
         "📈 Statistical Analysis", 
         "📉 Visualizations", 
         "🤖 Prediction Model", 
         "⚙️ Data Settings"]
    )
    
    st.markdown("---")
    
    # File uploader in sidebar
    st.markdown("### 📁 Upload Data")
    uploaded_file = st.file_uploader(
        "Choose a CSV file", 
        type=['csv'],
        help="Upload BankChurners.csv file"
    )
    
    st.markdown("---")
    
    # Information
    with st.expander("ℹ️ About This App"):
        st.info("""
        **App Features:**
        1. Data loading and exploration
        2. Advanced statistical analysis
        3. Interactive visualizations
        4. Churn prediction model
        5. Detailed reporting
        
        **Required Data:**
        - Attrition_Flag
        - Customer_Age
        - Gender
        - Credit_Limit
        - Total_Trans_Amt
        - Total_Trans_Ct
        """)

# ===========================================
# 📥 Data Loading Function
# ===========================================
@st.cache_data
def load_data(uploaded_file=None):
    """
    Load data from uploaded file or local file
    """
    try:
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.session_state['data_source'] = "Uploaded File"
            return df
        
        # Try local paths
        paths = [
            "BankChurners.csv",
            "./BankChurners.csv",
            "data/BankChurners.csv",
            "../BankChurners.csv"
        ]
        
        for path in paths:
            try:
                df = pd.read_csv(path)
                st.session_state['data_source'] = f"Local File: {path}"
                return df
            except:
                continue
        
        # Create sample data if no file found
        st.warning("⚠️ Using sample data. Please upload your data file.")
        np.random.seed(42)
        
        n_samples = 2000
        data = {
            'CLIENTNUM': range(100000, 100000 + n_samples),
            'Attrition_Flag': np.random.choice(['Existing Customer', 'Attrited Customer'], 
                                              n_samples, p=[0.85, 0.15]),
            'Customer_Age': np.random.randint(26, 70, n_samples),
            'Gender': np.random.choice(['M', 'F'], n_samples, p=[0.52, 0.48]),
            'Education_Level': np.random.choice(['Graduate', 'High School', 'Unknown', 
                                                'Uneducated', 'College', 'Post-Graduate', 
                                                'Doctorate'], n_samples),
            'Marital_Status': np.random.choice(['Married', 'Single', 'Divorced', 'Unknown'], 
                                              n_samples),
            'Income_Category': np.random.choice(['Less than $40K', '$40K - $60K', 
                                                '$60K - $80K', '$80K - $120K', 
                                                '$120K +', 'Unknown'], n_samples),
            'Credit_Limit': np.random.normal(10000, 4000, n_samples).astype(int),
            'Total_Revolving_Bal': np.random.normal(2000, 800, n_samples).astype(int),
            'Avg_Open_To_Buy': np.random.normal(8000, 3500, n_samples).astype(int),
            'Total_Trans_Amt': np.random.normal(5000, 2000, n_samples).astype(int),
            'Total_Trans_Ct': np.random.normal(60, 20, n_samples).astype(int)
        }
        
        df = pd.DataFrame(data)
        # Ensure positive values
        for col in ['Credit_Limit', 'Total_Revolving_Bal', 'Total_Trans_Amt', 'Total_Trans_Ct']:
            df[col] = df[col].apply(lambda x: max(x, 0))
        
        st.session_state['data_source'] = "Sample Data"
        return df
        
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        return pd.DataFrame()

# ===========================================
# 🧹 Data Cleaning Function
# ===========================================
def clean_data(df):
    """
    Clean and prepare data for analysis
    """
    if df.empty:
        return df
    
    df_clean = df.copy()
    
    # Select important columns if they exist
    important_cols = []
    available_cols = df_clean.columns.tolist()
    
    target_cols = ['Attrition_Flag', 'Customer_Age', 'Gender', 
                   'Education_Level', 'Marital_Status', 'Income_Category',
                   'Credit_Limit', 'Total_Revolving_Bal', 
                   'Total_Trans_Amt', 'Total_Trans_Ct']
    
    for col in target_cols:
        if col in available_cols:
            important_cols.append(col)
    
    if len(important_cols) < 5:  # Minimum required columns
        st.error("Data does not contain required columns")
        return df_clean[available_cols[:10]]  # Return first 10 columns
    
    df_clean = df_clean[important_cols]
    
    # Clean and transform
    df_clean = df_clean.dropna().drop_duplicates()
    
    # Convert to categorical
    categorical_cols = ['Attrition_Flag', 'Gender', 'Education_Level', 
                       'Marital_Status', 'Income_Category']
    
    for col in categorical_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype('category')
    
    # Create churn numeric column
    if 'Attrition_Flag' in df_clean.columns:
        df_clean['Churn_Numeric'] = (df_clean['Attrition_Flag'] == 'Attrited Customer').astype(int)
    
    return df_clean

# ===========================================
# 📊 PAGE 1: Data Overview
# ===========================================
if page == "📊 Data Overview":
    st.markdown('<h2 class="section-header">📊 Data Overview</h2>', unsafe_allow_html=True)
    
    # Load data
    df_raw = load_data(uploaded_file)
    df = clean_data(df_raw)
    
    if not df.empty:
        # Display data source
        source = st.session_state.get('data_source', 'Unknown')
        st.success(f"✅ Data Source: {source}")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h4>👥 Total Customers</h4>
                <h2>{len(df):,}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            if 'Attrition_Flag' in df.columns:
                churned = (df['Attrition_Flag'] == 'Attrited Customer').sum()
                churn_rate = (churned / len(df)) * 100
                st.markdown(f"""
                <div class="metric-card">
                    <h4>📉 Churn Rate</h4>
                    <h2>{churn_rate:.1f}%</h2>
                    <small>{churned:,} churned customers</small>
                </div>
                """, unsafe_allow_html=True)
        
        with col3:
            if 'Customer_Age' in df.columns:
                avg_age = df['Customer_Age'].mean()
                st.markdown(f"""
                <div class="metric-card">
                    <h4>🎂 Average Age</h4>
                    <h2>{avg_age:.1f}</h2>
                    <small>years</small>
                </div>
                """, unsafe_allow_html=True)
        
        with col4:
            if 'Credit_Limit' in df.columns:
                avg_credit = df['Credit_Limit'].mean()
                st.markdown(f"""
                <div class="metric-card">
                    <h4>💳 Average Credit Limit</h4>
                    <h2>${avg_credit:,.0f}</h2>
                </div>
                """, unsafe_allow_html=True)
        
        # Data Preview
        st.subheader("🔍 Data Preview")
        
        tab1, tab2, tab3 = st.tabs(["Data Sample", "Descriptive Statistics", "Data Information"])
        
        with tab1:
            num_rows = st.slider("Number of rows to display", 5, 50, 10)
            st.dataframe(df.head(num_rows), use_container_width=True)
        
        with tab2:
            if df.select_dtypes(include=[np.number]).shape[1] > 0:
                st.dataframe(df.describe(), use_container_width=True)
            else:
                st.warning("No numerical columns in data")
        
        with tab3:
            col_info = pd.DataFrame({
                'Column': df.columns,
                'Data Type': df.dtypes.astype(str),
                'Non-Null Count': df.notnull().sum(),
                'Unique Values': [df[col].nunique() for col in df.columns]
            })
            st.dataframe(col_info, use_container_width=True)
        
        # Data Quality Check
        st.subheader("✅ Data Quality Check")
        
        col1, col2 = st.columns(2)
        
        with col1:
            missing_values = df.isnull().sum()
            missing_df = pd.DataFrame({
                'Column': missing_values.index,
                'Missing Values': missing_values.values,
                'Percentage': (missing_values.values / len(df)) * 100
            })
            missing_df = missing_df[missing_df['Missing Values'] > 0]
            
            if len(missing_df) > 0:
                st.warning(f"⚠️ {len(missing_df)} columns have missing values")
                st.dataframe(missing_df, use_container_width=True)
            else:
                st.success("🎉 No missing values")
        
        with col2:
            duplicates = df.duplicated().sum()
            if duplicates > 0:
                st.warning(f"⚠️ {duplicates} duplicate rows found")
            else:
                st.success("✅ No duplicate data")
            
            # Data types check
            object_cols = df.select_dtypes(include=['object']).columns.tolist()
            if object_cols:
                st.info(f"📝 {len(object_cols)} text columns that can be converted to categories")
        
        # Download cleaned data
        st.subheader("💾 Download Data")
        
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Cleaned Data (CSV)",
            data=csv,
            file_name="bank_data_cleaned.csv",
            mime="text/csv",
            help="Download cleaned data"
        )

# ===========================================
# 📈 PAGE 2: Statistical Analysis
# ===========================================
elif page == "📈 Statistical Analysis":
    st.markdown('<h2 class="section-header">📈 Statistical Analysis</h2>', unsafe_allow_html=True)
    
    df_raw = load_data(uploaded_file)
    df = clean_data(df_raw)
    
    if not df.empty and 'Attrition_Flag' in df.columns:
        # Chi-square Test
        st.subheader("📊 Chi-square Test: Relationship Between Gender and Churn")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Contingency table
            gender_table = pd.crosstab(df['Gender'], df['Attrition_Flag'], 
                                      margins=True, margins_name="Total")
            st.write("##### Contingency Table")
            st.dataframe(gender_table, use_container_width=True)
        
        with col2:
            # Chi-square calculation
            from scipy.stats import chi2_contingency
            
            chi2, p, dof, expected = chi2_contingency(
                pd.crosstab(df['Gender'], df['Attrition_Flag'])
            )
            
            st.markdown(f"""
            <div class="metric-card">
                <h4>📊 Chi-square Test Results</h4>
                <p><strong>Chi-square Value:</strong> {chi2:.4f}</p>
                <p><strong>P-value:</strong> {p:.6f}</p>
                <p><strong>Degrees of Freedom:</strong> {dof}</p>
                <p><strong>Statistical Significance (α=0.05):</strong> {"✅ Yes" if p < 0.05 else "❌ No"}</p>
            </div>
            """, unsafe_allow_html=True)
            
            if p < 0.05:
                st.success("✅ There is a statistically significant relationship between gender and churn")
            else:
                st.info("ℹ️ No evidence of statistically significant relationship between gender and churn")
        
        # T-tests
        st.subheader("📈 T-tests for Numerical Variables")
        
        # Select numerical variables
        numeric_vars = []
        for col in ['Customer_Age', 'Credit_Limit', 'Total_Trans_Amt', 
                   'Total_Trans_Ct', 'Total_Revolving_Bal']:
            if col in df.columns:
                numeric_vars.append(col)
        
        if numeric_vars:
            t_test_results = []
            
            for var in numeric_vars:
                existing = df[df['Attrition_Flag'] == 'Existing Customer'][var]
                attrited = df[df['Attrition_Flag'] == 'Attrited Customer'][var]
                
                t_stat, p_value = stats.ttest_ind(existing, attrited, equal_var=False)
                
                t_test_results.append({
                    'Variable': var,
                    'Mean (Existing)': f"{existing.mean():.2f}",
                    'Mean (Attrited)': f"{attrited.mean():.2f}",
                    'Difference': f"{existing.mean() - attrited.mean():.2f}",
                    'T-statistic': f"{t_stat:.4f}",
                    'P-value': f"{p_value:.6f}",
                    'Significant': '✅ Yes' if p_value < 0.05 else '❌ No'
                })
            
            t_test_df = pd.DataFrame(t_test_results)
            st.dataframe(t_test_df, use_container_width=True)
            
            # Highlight significant results
            significant_vars = [row['Variable'] for _, row in t_test_df.iterrows() 
                              if row['Significant'] == '✅ Yes']
            if significant_vars:
                st.success(f"✅ Statistically significant variables: {', '.join(significant_vars)}")
        
        # Correlation Analysis
        st.subheader("🔗 Correlation Analysis")
        
        if len(df.select_dtypes(include=[np.number]).columns) > 1:
            numeric_df = df.select_dtypes(include=[np.number])
            corr_matrix = numeric_df.corr()
            
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                       center=0, square=True, ax=ax, cbar_kws={"shrink": 0.8})
            ax.set_title('Correlation Matrix of Numerical Variables', fontsize=14)
            st.pyplot(fig)
            
            # Strongest correlations
            st.write("##### Strongest Correlations")
            corr_pairs = corr_matrix.unstack().sort_values(ascending=False)
            # Remove self-correlations
            corr_pairs = corr_pairs[corr_pairs.index.get_level_values(0) != corr_pairs.index.get_level_values(1)]
            top_corrs = corr_pairs.head(10)
            
            for (var1, var2), value in top_corrs.items():
                col1, col2, col3 = st.columns([2, 2, 1])
                with col1:
                    st.write(f"**{var1}**")
                with col2:
                    st.write(f"**{var2}**")
                with col3:
                    st.write(f"**{value:.3f}**")

# ===========================================
# 📉 PAGE 3: Visualizations
# ===========================================
elif page == "📉 Visualizations":
    st.markdown('<h2 class="section-header">📉 Data Visualizations</h2>', unsafe_allow_html=True)
    
    df_raw = load_data(uploaded_file)
    df = clean_data(df_raw)
    
    if not df.empty and 'Attrition_Flag' in df.columns:
        # Visualization controls
        col1, col2 = st.columns(2)
        
        with col1:
            chart_type = st.selectbox(
                "Select Chart Type:",
                ["Churn Distribution", "Churn by Gender", "Age Distribution", 
                 "Credit Limit", "Transactions", "Variable Relationships"]
            )
        
        with col2:
            color_scheme = st.selectbox(
                "Select Color Scheme:",
                ["Green/Red", "Blue/Orange", "Black/Gray", "Purple/Gold"]
            )
            
            # Set colors based on scheme
            if color_scheme == "Green/Red":
                colors = ['#2E8B57', '#DC143C']
            elif color_scheme == "Blue/Orange":
                colors = ['#4682B4', '#FF8C00']
            elif color_scheme == "Black/Gray":
                colors = ['#404040', '#808080']
            else:  # Purple/Gold
                colors = ['#6A5ACD', '#FFD700']
        
        # Create visualizations
        if chart_type == "Churn Distribution":
            fig = px.pie(df, names='Attrition_Flag',
                        title='Customer Status Distribution',
                        color='Attrition_Flag',
                        color_discrete_sequence=colors,
                        hole=0.3)
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
            
            # Add bar chart
            fig2 = px.bar(df['Attrition_Flag'].value_counts().reset_index(),
                         x='Attrition_Flag', y='count',
                         title='Customer Count by Status',
                         color='Attrition_Flag',
                         color_discrete_sequence=colors)
            st.plotly_chart(fig2, use_container_width=True)
        
        elif chart_type == "Churn by Gender":
            gender_churn = pd.crosstab(df['Gender'], df['Attrition_Flag'])
            gender_churn_pct = gender_churn.div(gender_churn.sum(axis=1), axis=0) * 100
            
            fig = px.bar(gender_churn_pct.reset_index().melt(id_vars='Gender'),
                        x='Gender', y='value', color='Attrition_Flag',
                        title='Churn Rate by Gender (%)',
                        barmode='group',
                        color_discrete_sequence=colors,
                        labels={'value': 'Percentage'})
            st.plotly_chart(fig, use_container_width=True)
        
        elif chart_type == "Age Distribution":
            fig = px.histogram(df, x='Customer_Age', color='Attrition_Flag',
                              title='Age Distribution by Churn Status',
                              color_discrete_sequence=colors,
                              marginal='box',
                              opacity=0.7,
                              nbins=30)
            st.plotly_chart(fig, use_container_width=True)
        
        elif chart_type == "Credit Limit":
            fig = px.box(df, x='Attrition_Flag', y='Credit_Limit',
                        title='Credit Limit by Churn Status',
                        color='Attrition_Flag',
                        color_discrete_sequence=colors)
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
            # Add violin plot
            fig2 = px.violin(df, x='Attrition_Flag', y='Credit_Limit',
                            title='Credit Limit Density',
                            color='Attrition_Flag',
                            color_discrete_sequence=colors,
                            box=True)
            st.plotly_chart(fig2, use_container_width=True)
        
        elif chart_type == "Transactions":
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.box(df, x='Attrition_Flag', y='Total_Trans_Amt',
                            title='Transaction Amount',
                            color='Attrition_Flag',
                            color_discrete_sequence=colors)
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.box(df, x='Attrition_Flag', y='Total_Trans_Ct',
                            title='Transaction Count',
                            color='Attrition_Flag',
                            color_discrete_sequence=colors)
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
        
        elif chart_type == "Variable Relationships":
            fig = px.scatter(df, x='Total_Trans_Ct', y='Total_Trans_Amt',
                            color='Attrition_Flag',
                            title='Relationship Between Transaction Count and Amount',
                            color_discrete_sequence=colors,
                            opacity=0.6,
                            size='Customer_Age' if 'Customer_Age' in df.columns else None,
                            hover_data=['Gender', 'Credit_Limit'])
            st.plotly_chart(fig, use_container_width=True)
        
        # Interactive filters
        st.subheader("🔍 Interactive Data Filters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'Customer_Age' in df.columns:
                age_range = st.slider(
                    "Age Range:",
                    int(df['Customer_Age'].min()),
                    int(df['Customer_Age'].max()),
                    (int(df['Customer_Age'].min()), int(df['Customer_Age'].max()))
                )
        
        with col2:
            if 'Credit_Limit' in df.columns:
                credit_range = st.slider(
                    "Credit Limit Range:",
                    int(df['Credit_Limit'].min()),
                    int(df['Credit_Limit'].max()),
                    (int(df['Credit_Limit'].min()), int(df['Credit_Limit'].max()))
                )
        
        # Apply filters
        filtered_df = df.copy()
        if 'Customer_Age' in df.columns:
            filtered_df = filtered_df[
                (filtered_df['Customer_Age'] >= age_range[0]) & 
                (filtered_df['Customer_Age'] <= age_range[1])
            ]
        
        if 'Credit_Limit' in df.columns:
            filtered_df = filtered_df[
                (filtered_df['Credit_Limit'] >= credit_range[0]) & 
                (filtered_df['Credit_Limit'] <= credit_range[1])
            ]
        
        st.info(f"📊 Number of customers after filtering: {len(filtered_df):,}")

# ===========================================
# 🤖 PAGE 4: Prediction Model
# ===========================================
elif page == "🤖 Prediction Model":
    st.markdown('<h2 class="section-header">🤖 Churn Prediction Model</h2>', unsafe_allow_html=True)
    
    df_raw = load_data(uploaded_file)
    df = clean_data(df_raw)
    
    if not df.empty and 'Churn_Numeric' in df.columns:
        st.info("""
        **Note:** This model uses Logistic Regression to predict customer churn probability.
        Data is split into 80% training and 20% testing.
        """)
        
        # Model configuration
        st.subheader("⚙️ Model Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Feature selection
            available_features = [col for col in df.columns 
                                if col not in ['Attrition_Flag', 'Churn_Numeric'] 
                                and df[col].dtype in [np.int64, np.float64, 'category']]
            
            selected_features = st.multiselect(
                "Select Independent Variables:",
                available_features,
                default=['Customer_Age', 'Credit_Limit', 'Total_Trans_Amt', 'Total_Trans_Ct']
            )
        
        with col2:
            test_size = st.slider("Test Data Size (%)", 10, 40, 20)
            random_state = st.number_input("Random Seed", 0, 100, 42)
        
        if len(selected_features) > 0:
            # Prepare data
            X = df[selected_features].copy()
            y = df['Churn_Numeric']
            
            # Encode categorical variables
            X_encoded = pd.get_dummies(X, drop_first=True)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_encoded, y, 
                test_size=test_size/100, 
                random_state=random_state, 
                stratify=y
            )
            
            # Train model
            model = LogisticRegression(max_iter=1000, random_state=random_state)
            model.fit(X_train, y_train)
            
            # Evaluate model
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Display results
            st.subheader("📊 Model Performance")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                accuracy = accuracy_score(y_test, y_pred)
                st.markdown(f"""
                <div class="metric-card">
                    <h4>Accuracy</h4>
                    <h2>{accuracy*100:.2f}%</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                from sklearn.metrics import precision_score
                precision = precision_score(y_test, y_pred)
                st.markdown(f"""
                <div class="metric-card">
                    <h4>Precision</h4>
                    <h2>{precision*100:.2f}%</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                from sklearn.metrics import recall_score
                recall = recall_score(y_test, y_pred)
                st.markdown(f"""
                <div class="metric-card">
                    <h4>Recall</h4>
                    <h2>{recall*100:.2f}%</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                from sklearn.metrics import f1_score
                f1 = f1_score(y_test, y_pred)
                st.markdown(f"""
                <div class="metric-card">
                    <h4>F1 Score</h4>
                    <h2>{f1*100:.2f}%</h2>
                </div>
                """, unsafe_allow_html=True)
            
            # Confusion Matrix
            st.subheader("📈 Confusion Matrix")
            
            cm = confusion_matrix(y_test, y_pred)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=['Not Churn', 'Churn'],
                       yticklabels=['Not Churn', 'Churn'])
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_title('Confusion Matrix')
            st.pyplot(fig)
            
            # Feature Importance
            st.subheader("🔝 Feature Importance")
            
            if len(model.coef_) > 0:
                feature_importance = pd.DataFrame({
                    'Feature': X_encoded.columns,
                    'Coefficient': model.coef_[0],
                    'Odds Ratio': np.exp(model.coef_[0])
                }).sort_values('Odds Ratio', ascending=False)
                
                fig = px.bar(feature_importance.head(15), 
                            x='Odds Ratio', 
                            y='Feature',
                            orientation='h',
                            title='Top 15 Features (Odds Ratio)',
                            color='Odds Ratio',
                            color_continuous_scale='RdYlBu_r')
                st.plotly_chart(fig, use_container_width=True)
            
            # Prediction Interface
            st.subheader("🔮 New Prediction")
            
            with st.form("prediction_form"):
                st.write("Enter customer data for prediction:")
                
                col1, col2 = st.columns(2)
                
                input_data = {}
                with col1:
                    if 'Customer_Age' in selected_features:
                        input_data['Customer_Age'] = st.number_input("Age", 18, 100, 45)
                    
                    if 'Credit_Limit' in selected_features:
                        input_data['Credit_Limit'] = st.number_input("Credit Limit", 0, 50000, 10000)
                
                with col2:
                    if 'Total_Trans_Amt' in selected_features:
                        input_data['Total_Trans_Amt'] = st.number_input("Transaction Amount", 0, 20000, 5000)
                    
                    if 'Total_Trans_Ct' in selected_features:
                        input_data['Total_Trans_Ct'] = st.number_input("Transaction Count", 0, 200, 60)
                
                predict_btn = st.form_submit_button("Predict")
                
                if predict_btn:
                    # Create input dataframe
                    input_df = pd.DataFrame([input_data])
                    
                    # Align with training data
                    input_encoded = pd.get_dummies(input_df)
                    
                    for col in X_encoded.columns:
                        if col not in input_encoded.columns:
                            input_encoded[col] = 0
                    
                    input_encoded = input_encoded[X_encoded.columns]
                    
                    # Make prediction
                    probability = model.predict_proba(input_encoded)[0, 1]
                    
                    # Display result
                    if probability > 0.5:
                        st.error(f"⚠️ **Prediction: Customer will churn** (probability: {probability:.1%})")
                    else:
                        st.success(f"✅ **Prediction: Customer will stay** (probability: {1-probability:.1%})")
                    
                    # Visual gauge
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=probability * 100,
                        title={'text': "Churn Probability"},
                        domain={'x': [0, 1], 'y': [0, 1]},
                        gauge={
                            'axis': {'range': [0, 100]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 30], 'color': "green"},
                                {'range': [30, 70], 'color': "yellow"},
                                {'range': [70, 100], 'color': "red"}
                            ],
                            'threshold': {
                                'line': {'color': "black", 'width': 4},
                                'thickness': 0.75,
                                'value': 50
                            }
                        }
                    ))
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)

# ===========================================
# ⚙️ PAGE 5: Data Settings
# ===========================================
elif page == "⚙️ Data Settings":
    st.markdown('<h2 class="section-header">⚙️ Data Settings & Management</h2>', unsafe_allow_html=True)
    
    df_raw = load_data(uploaded_file)
    
    if not df_raw.empty:
        st.subheader("🔧 Data Processing")
        
        # Data cleaning options
        cleaning_options = st.multiselect(
            "Select cleaning operations:",
            ["Remove Missing Values", "Remove Duplicates", 
             "Select Important Columns", "Convert Data Types"]
        )
        
        if st.button("🔧 Apply Cleaning"):
            df_clean = df_raw.copy()
            
            if "Remove Missing Values" in cleaning_options:
                initial_rows = len(df_clean)
                df_clean = df_clean.dropna()
                st.info(f"Removed {initial_rows - len(df_clean)} rows with missing values")
            
            if "Remove Duplicates" in cleaning_options:
                initial_rows = len(df_clean)
                df_clean = df_clean.drop_duplicates()
                st.info(f"Removed {initial_rows - len(df_clean)} duplicate rows")
            
            if "Select Important Columns" in cleaning_options:
                important_cols = st.multiselect(
                    "Select important columns:",
                    df_clean.columns.tolist(),
                    default=['Attrition_Flag', 'Customer_Age', 'Gender', 
                            'Credit_Limit', 'Total_Trans_Amt', 'Total_Trans_Ct']
                )
                if important_cols:
                    df_clean = df_clean[important_cols]
            
            # Save cleaned data
            st.session_state['cleaned_data'] = df_clean
            
            st.success("✅ Data cleaned successfully!")
            st.dataframe(df_clean.head(), use_container_width=True)
        
        # Data export
        st.subheader("💾 Export Data")
        
        if 'cleaned_data' in st.session_state:
            df_to_export = st.session_state['cleaned_data']
        else:
            df_to_export = df_raw
        
        export_format = st.radio(
            "Select export format:",
            ["CSV", "Excel", "JSON"]
        )
        
        if export_format == "CSV":
            csv = df_to_export.to_csv(index=False)
            st.download_button(
                label="📥 Download as CSV",
                data=csv,
                file_name="bank_data_export.csv",
                mime="text/csv"
            )
        
        elif export_format == "Excel":
            # For Excel export, we need to use BytesIO
            import io
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                df_to_export.to_excel(writer, index=False, sheet_name='Bank_Data')
            
            st.download_button(
                label="📥 Download as Excel",
                data=buffer,
                file_name="bank_data_export.xlsx",
                mime="application/vnd.ms-excel"
            )
        
        # Data summary
        st.subheader("📋 Data Summary")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**General Information:**")
            st.write(f"- Number of rows: {len(df_to_export):,}")
            st.write(f"- Number of columns: {len(df_to_export.columns)}")
            st.write(f"- Memory usage: {df_to_export.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        with col2:
            st.write("**Data Types:**")
            type_counts = df_to_export.dtypes.value_counts()
            for dtype, count in type_counts.items():
                st.write(f"- {dtype}: {count} columns")

# ===========================================
# 🏁 Footer
# ===========================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem 0;'>
    <p><strong>🏦 Bank Customer Churn Analysis Application</strong></p>
    <p>Developed with Streamlit | For educational and analytical purposes</p>
    <p>© 2024 All rights reserved</p>
</div>
""", unsafe_allow_html=True)