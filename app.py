import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import random

random.seed(42)
np.random.seed(42)

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.fuzzy_logic import FuzzyStressPredictor
from src.genetic_algorithm import GeneticAlgorithmOptimizer
import joblib

# Page configuration
st.set_page_config(
    page_title="Adaptive Stress Predictor",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #1E88E5;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load or train the model"""
    model_path = 'models/fuzzy_ga_model.pkl'
    data_path = 'data/stress_detection_data.csv'
    
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        # Train model if not exists
        with st.spinner("Training model for the first time... This may take a minute."):
            df = pd.read_csv(data_path)
            ga = GeneticAlgorithmOptimizer(df)
            best_weights = ga.evolve()
            predictor = FuzzyStressPredictor(best_weights)
            
            # Save model
            os.makedirs('models', exist_ok=True)
            joblib.dump(predictor, model_path)
            
        return predictor

def main():
    # Header
    st.markdown('<h1 class="main-header">Adaptive Stress Prediction System</h1>', unsafe_allow_html=True)
    st.markdown("### Using Fuzzy Logic Controlled Genetic Algorithms")
    
    # Sidebar for training
    with st.sidebar:
        st.header("⚙️ Model Settings")
        
        if st.button("🔄 Retrain Model"):
            if os.path.exists('models/fuzzy_ga_model.pkl'):
                os.remove('models/fuzzy_ga_model.pkl')
            st.rerun()
        
        st.markdown("---")
        st.markdown("### About")
        st.info("""
        This system uses:
        - **Fuzzy Logic**: Handle uncertainty in lifestyle data
        - **Genetic Algorithms**: Optimize fuzzy rule weights
        - **Real-time Prediction**: Live stress prediction based on User Input
        """)
    
    # Load model
    predictor = load_model()
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["📝 Predict Stress", "📊 Batch Prediction", "ℹ️ Info"])
    
    with tab1:
        st.header("Enter Your Lifestyle Information")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("👤 Personal")
            age = st.slider("Age", 18, 70, 30)
            gender = st.selectbox("Gender", ["Male", "Female"])
            marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
            
            st.subheader("😴 Sleep")
            sleep_duration = st.slider("Sleep Duration (hours)", 3.0, 10.0, 7.0, 0.5)
            sleep_quality = st.slider("Sleep Quality (1-5)", 1, 5, 4)
            
        with col2:
            st.subheader("💼 Work & Activity")
            work_hours = st.slider("Work Hours/day", 6, 14, 8)
            travel_time = st.slider("Travel Time (hours)", 0.0, 5.0, 1.0, 0.5)
            physical_activity = st.slider("Physical Activity (hours/week)", 0.0, 7.0, 2.0, 0.5)
            social_interactions = st.slider("Social Interactions/week", 1, 7, 4)
            
            st.subheader("🧘 Wellness")
            meditation = st.selectbox("Meditation Practice", ["Yes", "No"])
            
        with col3:
            st.subheader("📱 Screen & Habits")
            screen_time = st.slider("Screen Time (hours/day)", 1, 8, 4)
            caffeine_intake = st.slider("Caffeine (cups/day)", 0, 5, 1)
            alcohol_intake = st.slider("Alcohol (drinks/week)", 0, 5, 0)
            smoking = st.selectbox("Smoking Habit", ["Yes", "No"])
            
            st.subheader("🏥 Health Metrics")
            blood_pressure = st.slider("Blood Pressure (systolic)", 90, 180, 120)
            cholesterol = st.slider("Cholesterol (mg/dL)", 150, 300, 200)
            blood_sugar = st.slider("Blood Sugar (mg/dL)", 70, 150, 90)
        
        # Predict button
        if st.button("Predict Stress Level", type="primary"):
            # Prepare input data
            input_data = {
                'Age': age,
                'Gender': 1 if gender == "Male" else 0,
                'Marital_Status': ["Single", "Married", "Divorced"].index(marital_status),
                'Sleep_Duration': sleep_duration,
                'Sleep_Quality': sleep_quality,
                'Physical_Activity': physical_activity,
                'Screen_Time': screen_time,
                'Caffeine_Intake': caffeine_intake,
                'Alcohol_Intake': alcohol_intake,
                'Smoking_Habit': 1 if smoking == "Yes" else 0,
                'Work_Hours': work_hours,
                'Travel_Time': travel_time,
                'Social_Interactions': social_interactions,
                'Meditation_Practice': 1 if meditation == "Yes" else 0,
                'Blood_Pressure': blood_pressure,
                'Cholesterol_Level': cholesterol,
                'Blood_Sugar_Level': blood_sugar
            }
            
            # Predict
            stress_level, confidence = predictor.predict(input_data)
            
            # Display results
            st.markdown("---")
            st.header("📊 Prediction Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Stress level with color coding
                if stress_level == "Low":
                    st.success(f"### Stress Level: {stress_level} 😊")
                elif stress_level == "Medium":
                    st.warning(f"### Stress Level: {stress_level} 😐")
                else:
                    st.error(f"### Stress Level: {stress_level} 😟")
            
            with col2:
                st.metric("Confidence", f"{confidence:.1f}%")
            
            # Recommendations
            st.subheader("💡 Recommendations")
            recommendations = predictor.get_recommendations(input_data, stress_level)
            for rec in recommendations:
                st.write(f"• {rec}")
    
    with tab2:
        st.header("Batch Prediction from CSV")
        
        uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.write("### Preview:", df.head())
            
            if st.button("Run Predictions"):
                with st.spinner("Processing..."):
                    # Preprocess the dataframe
                    df_processed = df.copy()
                    
                    # Convert categorical columns to numeric
                    if 'Gender' in df_processed.columns:
                        df_processed['Gender'] = df_processed['Gender'].map({'Male': 1, 'Female': 0})
                    if 'Marital_Status' in df_processed.columns:
                        df_processed['Marital_Status'] = df_processed['Marital_Status'].map({
                            'Single': 0, 'Married': 1, 'Divorced': 2
                        })
                    if 'Smoking_Habit' in df_processed.columns:
                        df_processed['Smoking_Habit'] = df_processed['Smoking_Habit'].map({'Yes': 1, 'No': 0})
                    if 'Meditation_Practice' in df_processed.columns:
                        df_processed['Meditation_Practice'] = df_processed['Meditation_Practice'].map({'Yes': 1, 'No': 0})
                    
                    predictions = []
                    for _, row in df_processed.iterrows():
                        input_dict = row.to_dict()
                        stress, conf = predictor.predict(input_dict)
                        predictions.append({'Stress_Level': stress, 'Confidence': conf})
                    
                    result_df = pd.concat([df, pd.DataFrame(predictions)], axis=1)
                    st.success("✅ Predictions completed!")
                    st.dataframe(result_df)
                    
                    # Download button
                    csv = result_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results",
                        data=csv,
                        file_name="stress_predictions.csv",
                        mime="text/csv"
                    )

    with tab3:
        st.header("System Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🧬 Fuzzy Logic")
            st.markdown("""
            **Membership Functions:**
            - Sleep: Low, Medium, High
            - Work: Normal, High, Excessive
            - Activity: Low, Medium, High
            - Health Risk: Low, Medium, High
            
            **Fuzzy Rules:**
            - IF sleep is Low AND work is High THEN stress is High
            - IF activity is High AND meditation is Yes THEN stress is Low
            - Over 10+ optimized rules
            """)
        
        with col2:
            st.subheader("🧬 Genetic Algorithm")
            st.markdown("""
            **Parameters:**
            - Population Size: 100
            - Generations: 50
            - Mutation Rate: 0.1
            - Crossover Rate: 0.8
            
            **Optimization:**
            - Evolves fuzzy rule weights
            - Maximizes prediction accuracy
            - Balances rule complexity
            """)
        
        st.subheader("📈 Model Performance")
        st.info("""
        The model is trained on around 770 samples with lifestyle and health data.
        """)

       
        st.markdown(
            """
            <style>
            .footer {
                position: fixed;
                left: 0;
                bottom: 0;
                width: 100%;
                background-color: transparent;
                color: grey;
                text-align: center;
                padding: 10px 0;
                font-size: 18px;
            }
            </style>
            <div class="footer">
                © 2025 GauravDT  |  ⚠️ Disclaimer: This project is for reference only. In cases of extreme stress, please consult a healthcare professional.
            </div>
            """,
            unsafe_allow_html=True
        )

if __name__ == "__main__":
    main()