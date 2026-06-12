import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(page_title="Diabetes Prediction", layout="wide", initial_sidebar_state="expanded")

# Custom CSS
st.markdown("""
    <style>
        .main {
            padding-top: 2rem;
        }
        .metric-card {
            background-color: #f0f2f6;
            padding: 1.5rem;
            border-radius: 10px;
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("🏥 Diabetes Prediction Model")
st.markdown("---")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv('Diabetes_data.csv')
    return df

df = load_data()

# Sidebar Navigation
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio("Select Page", 
    ["Home", "Dataset Overview", "Make Prediction", "Model Performance", "About"])

# ============== PAGE 1: HOME ==============
if page == "Home":
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("""
        ### 🎯 Project Objective
        This ML model predicts whether a person has diabetes based on medical parameters.
        
        ### 📌 Features Used:
        - Pregnancies
        - Glucose Level
        - Blood Pressure
        - Skin Thickness
        - Insulin Level
        - Body Mass Index (BMI)
        - Diabetes Pedigree Function
        - Age
        """)
    
    with col2:
        st.info("""
        #### 📈 Quick Stats
        - **Total Records**: 768
        - **Positive Cases**: 268 (35%)
        - **Negative Cases**: 500 (65%)
        - **Model Used**: Random Forest
        """)

# ============== PAGE 2: DATASET OVERVIEW ==============
elif page == "Dataset Overview":
    st.header("📊 Dataset Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Dataset Preview")
        st.dataframe(df.head(10), use_container_width=True)
    
    with col2:
        st.subheader("Basic Statistics")
        st.dataframe(df.describe(), use_container_width=True)
    
    st.subheader("Missing Values")
    missing = df.isnull().sum()
    st.write(f"✅ No missing values: {missing.sum() == 0}")
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Outcome Distribution")
        fig, ax = plt.subplots(figsize=(8, 5))
        outcome_counts = df['Outcome'].value_counts()
        colors = ['#FF6B6B', '#4ECDC4']
        ax.bar(['No Diabetes (0)', 'Diabetes (1)'], outcome_counts.values, color=colors)
        ax.set_ylabel('Count')
        ax.set_title('Outcome Distribution')
        st.pyplot(fig)
    
    with col2:
        st.subheader("Correlation Matrix")
        fig, ax = plt.subplots(figsize=(10, 8))
        correlation = df.corr()
        sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm', ax=ax, cbar_kws={'label': 'Correlation'})
        st.pyplot(fig)

# ============== PAGE 3: MAKE PREDICTION ==============
elif page == "Make Prediction":
    st.header("🔮 Make a Prediction")
    st.markdown("Enter patient medical parameters to predict diabetes risk:")
    
    # Create input columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=0)
        insulin = st.number_input("Insulin", min_value=0, max_value=900, value=0)
    
    with col2:
        glucose = st.number_input("Glucose", min_value=0, max_value=300, value=100)
        bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
    
    with col3:
        blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=70)
        dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
    
    with col4:
        skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
        age = st.number_input("Age", min_value=0, max_value=120, value=30)
    
    # Prediction button
    if st.button("🎯 Predict", key="predict_btn"):
        # Train model on full data
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_scaled, y)
        
        # Make prediction
        input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, 
                               insulin, bmi, dpf, age]])
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)
        probability = model.predict_proba(input_scaled)
        
        # Display results
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            if prediction[0] == 0:
                st.success("✅ **PREDICTION: No Diabetes**", icon="✓")
                st.metric("Confidence", f"{probability[0][0]*100:.2f}%")
            else:
                st.error("⚠️ **PREDICTION: Diabetes Risk Detected**", icon="✗")
                st.metric("Confidence", f"{probability[0][1]*100:.2f}%")
        
        with col2:
            # Probability visualization
            fig, ax = plt.subplots(figsize=(8, 5))
            categories = ['No Diabetes', 'Diabetes']
            probs = [probability[0][0]*100, probability[0][1]*100]
            colors_prob = ['#4ECDC4', '#FF6B6B']
            bars = ax.barh(categories, probs, color=colors_prob)
            ax.set_xlabel('Probability (%)')
            ax.set_xlim(0, 100)
            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax.text(width, bar.get_y() + bar.get_height()/2, 
                       f'{probs[i]:.1f}%', ha='left', va='center', fontweight='bold')
            st.pyplot(fig)
        
        st.info(f"""
        ### 📋 Input Summary
        - **Pregnancies**: {pregnancies}
        - **Glucose**: {glucose} mg/dL
        - **Blood Pressure**: {blood_pressure} mmHg
        - **Skin Thickness**: {skin_thickness} mm
        - **Insulin**: {insulin} mIU/L
        - **BMI**: {bmi}
        - **Diabetes Pedigree Function**: {dpf}
        - **Age**: {age} years
        """)

# ============== PAGE 4: MODEL PERFORMANCE ==============
elif page == "Model Performance":
    st.header("📊 Model Performance Metrics")
    
    # Train model
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Accuracy Score", f"{accuracy*100:.2f}%")
    
    with col2:
        st.metric("Training Samples", len(X_train))
    
    with col3:
        st.metric("Testing Samples", len(X_test))
    
    st.markdown("---")
    
    # Confusion Matrix
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, 
                   xticklabels=['No Diabetes', 'Diabetes'],
                   yticklabels=['No Diabetes', 'Diabetes'], ax=ax)
        ax.set_ylabel('Actual')
        ax.set_xlabel('Predicted')
        st.pyplot(fig)
    
    with col2:
        st.subheader("Classification Report")
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df, use_container_width=True)
    
    # Feature Importance
    st.subheader("📊 Feature Importance")
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(feature_importance['Feature'], feature_importance['Importance'], color='#FF6B6B')
    ax.set_xlabel('Importance Score')
    ax.set_title('Feature Importance in Diabetes Prediction')
    st.pyplot(fig)

# ============== PAGE 5: ABOUT ==============
elif page == "About":
    st.header("ℹ️ About This Project")
    
    st.markdown("""
    ### 📚 Project Information
    - **Dataset**: Diabetes Prediction Dataset
    - **Total Records**: 768 patients
    - **Features**: 8 medical parameters
    - **Problem Type**: Binary Classification
    
    ### 🤖 Models Implemented
    1. K-Nearest Neighbors (KNN)
    2. Logistic Regression
    3. Random Forest Classifier
    4. Support Vector Machine (SVM)
    5. Decision Tree Classifier
    
    ### 📊 Best Model: Random Forest
    - Highest accuracy among all models
    - Good generalization
    - Handles feature interactions well
    
    ### 👨‍💻 Developer
    Data Science Student - Semester 4
    
    ### ⚠️ Disclaimer
    This model is for educational purposes only. Always consult with a healthcare professional 
    for medical diagnosis and treatment.
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray; font-size: 12px;">
    <p>🏥 Diabetes Prediction System | Educational Project</p>
    <p>© 2024 Data Science Semester Project</p>
</div>
""", unsafe_allow_html=True)
