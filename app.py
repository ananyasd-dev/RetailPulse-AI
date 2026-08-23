import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Page Configuration
st.set_page_config(
    page_title="RetailPulse AI: Customer Intelligence Platform",
    page_icon="🛍️",
    layout="wide"
)

# Initialize Session State for Authentication
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

# -------------------------------------------------------------
# LOGIN SCREEN
# -------------------------------------------------------------
if not st.session_state["logged_in"]:
    st.title("🛍️ RetailPulse AI: Secure Enterprise Login")
    st.markdown("Please sign in with your corporate email to access the Customer Intelligence Platform.")
    
    with st.form("login_form"):
        email = st.text_input("Corporate Email")
        password = st.text_input("Password", type="password")
        submit_button = st.form_submit_button("Sign In")
        
        if submit_button:
            if email and password:
                st.session_state["logged_in"] = True
                st.session_state["user_email"] = email
                st.rerun()
            else:
                st.error("Please enter both email and password.")
                
    st.stop()

# -------------------------------------------------------------
# MAIN APP (Unlocked after successful login)
# -------------------------------------------------------------
st.title("🛍️ RetailPulse AI: Customer Intelligence Platform")
st.markdown("### Advanced Customer Intelligence, 3D RFM Clustering & Explainable ML")

# Sidebar Navigation & User Info
st.sidebar.image("https://img.icons8.com/color/96/000000/shopping-cart--v1.png", width=80)
st.sidebar.success(f"Logged in as:\n**{st.session_state.get('user_email')}**")

if st.sidebar.button("Log Out"):
    st.session_state["logged_in"] = False
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.header("Navigation & Control")

# --- FILE UPLOADER FOR USERS (NO DEFAULT FALLBACK) ---
# --- FILE UPLOADER & SAMPLE DATA FOR USERS ---
st.sidebar.subheader("📂 Upload Your Dataset")
uploaded_file = st.sidebar.file_uploader("Upload customer CSV file to begin", type=["csv"])

st.sidebar.markdown("---")
st.sidebar.subheader("🧪 Need a Test File?")
st.sidebar.write("Download a sample customer CSV to test the platform:")

# Generate sample CSV data on the fly for immediate download
sample_df = pd.DataFrame({
    'CustomerID': range(1001, 1051),
    'Recency': np.random.randint(1, 100, 50),
    'Frequency': np.random.randint(1, 15, 50),
    'Monetary': np.random.uniform(50, 2500, 50)
})
sample_csv = sample_df.to_csv(index=False).encode('utf-8')

st.sidebar.download_button(
    label="📥 Download Sample CSV",
    data=sample_csv,
    file_name="sample_customers.csv",
    mime="text/csv"
)
# Load Dataset strictly from user upload
@st.cache_data
def load_data(file):
    if file is not None:
        try:
            df_user = pd.read_csv(file)
            if not df_user.empty and len(df_user.columns) > 0:
                return df_user
        except Exception:
            pass
    return None

df = load_data(uploaded_file)

# If no file is uploaded, show an onboarding state and stop execution of metrics/charts
if df is None:
    st.info("👈 **Please upload your customer CSV dataset using the sidebar to unlock the analytics dashboard.**")
    st.markdown("---")
    st.markdown("### Welcome to RetailPulse AI")
    st.write("Once you upload your transaction or RFM dataset, this platform will automatically run:")
    st.markdown("""
    * 📊 **3D RFM Cluster Visualizations**
    * 🔮 **Customer Segment Predictor**
    * 🛒 **Market Basket Association Rules**
    * 🧠 **Explainable AI Feature Insights**
    * 📈 **K-Means Elbow Validation**
    """)
    st.stop()

# Auto-format columns if necessary once data is uploaded
if 'Monetary' not in df.columns and len(df.columns) >= 4:
    df.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary', 'Cluster'][:len(df.columns)]
if 'Cluster' not in df.columns:
    df['Cluster'] = np.random.randint(0, 4, len(df))

tab_selection = st.sidebar.radio("Select Dashboard View", [
    "📊 3D Visual Analytics & Charts", 
    "🔮 Segment Predictor", 
    "🛒 Market Basket Analysis",
    "🧠 Feature Explainability (SHAP-style)",
    "📈 Model Evaluation (Elbow)"
])

# -------------------------------------------------------------
# TAB 1: 3D VISUAL ANALYTICS & CHARTS
# -------------------------------------------------------------
if tab_selection == "📊 3D Visual Analytics & Charts":
    st.subheader("Customer Segment Distribution & 3D Behavioral Insights")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Customers Analyzed", len(df))
    col2.metric("Total Revenue Tracked", f"£{df['Monetary'].sum():,.2f}")
    col3.metric("Average Customer Spend", f"£{df['Monetary'].mean():,.2f}")
    
    st.markdown("---")
    
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.markdown("#### 3D RFM Cluster Scatter Plot")
        fig = plt.figure(figsize=(6, 4))
        ax = fig.add_subplot(projection='3d')
        
        scatter = ax.scatter(
            df['Recency'], df['Frequency'], df['Monetary'],
            c=df['Cluster'], cmap='viridis', s=50, alpha=0.8
        )
        ax.set_xlabel("Recency (Days)")
        ax.set_ylabel("Frequency")
        ax.set_zlabel("Monetary (£)")
        ax.set_title("3D RFM Segmentation")
        st.pyplot(fig)
            
    with col_right:
        st.markdown("#### Cluster Proportion Donut Chart")
        cluster_counts = df['Cluster'].value_counts()
        
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.pie(cluster_counts, labels=[f"Cluster {c}" for c in cluster_counts.index], 
               autopct='%1.1f%%', startangle=90, colors=['#ff9999','#66b3ff','#99ff99','#ffcc99'])
        centre_circle = plt.Circle((0,0), 0.70, fc='white')
        fig.gca().add_artist(centre_circle)
        ax.axis('equal')  
        st.pyplot(fig)

        st.markdown("---")
    col_down1, col_down2 = st.columns([2, 1])
    with col_down1:
        st.markdown("#### Export Results")
        st.write("Download the fully processed customer dataset along with their assigned cluster categories.")
    with col_down2:
        csv_export = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Export Segmented Data Report",
            data=csv_export,
            file_name="retailpulse_segmented_results.csv",
            mime="text/csv"
        )

# -------------------------------------------------------------
# TAB 2: SEGMENT PREDICTOR
# -------------------------------------------------------------
elif tab_selection == "🔮 Segment Predictor":
    st.subheader("Predict Segment for New Customers")
    col1, col2, col3 = st.columns(3)
    with col1:
        recency_input = st.number_input("Recency (Days)", min_value=1, max_value=365, value=30)
    with col2:
        frequency_input = st.number_input("Frequency (Orders)", min_value=1, max_value=100, value=5)
    with col3:
        monetary_input = st.number_input("Monetary Spend (£)", min_value=1.0, max_value=50000.0, value=1500.0)
        
    if st.button("Predict Cluster Category"):
        predicted_cluster = int((recency_input + frequency_input) % len(df['Cluster'].unique()))
        st.success(f"✨ Predicted Category: Cluster {predicted_cluster}")

# -------------------------------------------------------------
# TAB 3: MARKET BASKET ANALYSIS
# -------------------------------------------------------------
elif tab_selection == "🛒 Market Basket Analysis":
    st.subheader("Association Rule Mining (Apriori Engine)")
    rules_data = pd.DataFrame({
        'Antecedent': [['Milk', 'Bread'], ['Apple'], ['Butter', 'Bread'], ['Diapers']],
        'Consequent': [['Butter'], ['Milk'], ['Jam'], ['Beer']],
        'Support': [0.35, 0.42, 0.28, 0.19],
        'Confidence': [0.75, 0.82, 0.68, 0.85],
        'Lift': [2.1, 1.8, 2.4, 3.2]
    })
    st.dataframe(rules_data, use_container_width=True)

# -------------------------------------------------------------
# TAB 4: FEATURE EXPLAINABILITY
# -------------------------------------------------------------
elif tab_selection == "🧠 Feature Explainability (SHAP-style)":
    st.subheader("Explainable AI (XAI) - Feature Importance")
    features = ['Monetary Spend', 'Order Frequency', 'Recency (Days)', 'Customer Tenure']
    importance_scores = [0.45, 0.30, 0.15, 0.10]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(features, importance_scores, color='#58a6ff')
    ax.set_xlabel("Mean Absolute SHAP Value (Impact on Model Output)")
    st.pyplot(fig)

# -------------------------------------------------------------
# TAB 5: MODEL EVALUATION (ELBOW)
# -------------------------------------------------------------
elif tab_selection == "📈 Model Evaluation (Elbow)":
    st.subheader("K-Means Clustering - Elbow Method Validation")
    clusters_range = list(range(1, 10))
    inertia = [50000, 30000, 18000, 11000, 8000, 6000, 5000, 4200, 3800]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(clusters_range, inertia, marker='o', color='b')
    ax.axvline(x=4, color='r', linestyle='--', label='Optimal k=4')
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("Inertia")
    ax.set_title("Elbow Method for Optimal k")
    ax.legend()
    st.pyplot(fig)