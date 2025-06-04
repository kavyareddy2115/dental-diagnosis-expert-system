import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier

# Set page config
st.set_page_config(
    page_title="Dental Diagnosis Expert System",
    layout="wide"
)

# Import the symptom mapping
from symptom_mapping import symptom_mapping, get_symptom_name

# Function to load or train model
@st.cache_resource
def load_or_train_model():
    # Define column names with descriptive labels
    names = list(symptom_mapping.values()) + ['Diagnosis']
    
    # Check if model exists
    if os.path.exists('dental_model.pkl'):
        with open('dental_model.pkl', 'rb') as file:
            return pickle.load(file)
    
    # If not, train model
    dataset = pd.read_excel('rm1.xls', names=names)
    
    # Data preprocessing
    X = dataset.iloc[:, 0:28]
    y = dataset.iloc[:, 28]
    
    # Normalize data
    scaler = MinMaxScaler()
    X_normalized = scaler.fit_transform(X)
    
    # Train model
    model = KNeighborsClassifier(n_neighbors=6, metric='euclidean')
    model.fit(X_normalized, y)
    
    # Save model data
    model_data = {
        'model': model,
        'scaler': scaler,
        'symptom_names': names[:-1],
        'dataset': dataset
    }
    
    with open('dental_model.pkl', 'wb') as file:
        pickle.dump(model_data, file)
    
    return model_data

# Load model data
model_data = load_or_train_model()
knn_model = model_data['model']
scaler = model_data['scaler']
symptom_names = model_data['symptom_names']
dataset = model_data.get('dataset')

# App title
st.title("Dental Diagnosis Expert System")

# Sidebar for navigation
page = st.sidebar.selectbox("Choose a page", ["Diagnosis Tool", "Model Information", "Data Visualization"])

if page == "Diagnosis Tool":
    st.header("Symptom Input")
    st.write("""
    Enter symptom values (0.0-1.0) where:
    - 0.0 = Not happening
    - 0.25 = Undecided
    - 0.5 = Maybe
    - 0.75 = Most likely
    - 1.0 = Sure
    """)
    
    # Create columns for symptom inputs
    col1, col2, col3 = st.columns(3)
    
    # Dictionary to store symptom values
    symptom_values = {}
    
    # Distribute symptoms across columns
    for i, symptom in enumerate(symptom_names):
        if i % 3 == 0:
            symptom_values[symptom] = col1.slider(
                f"{symptom}", 0.0, 1.0, 0.0, 0.25
            )
        elif i % 3 == 1:
            symptom_values[symptom] = col2.slider(
                f"{symptom}", 0.0, 1.0, 0.0, 0.25
            )
        else:
            symptom_values[symptom] = col3.slider(
                f"{symptom}", 0.0, 1.0, 0.0, 0.25
            )
    
    # Diagnosis button
    if st.button("Diagnose"):
        # Get symptom values as list
        symptoms_list = [symptom_values[symptom] for symptom in symptom_names]
        
        # Normalize input
        normalized_symptoms = scaler.transform([symptoms_list])
        
        # Make prediction
        prediction = knn_model.predict(normalized_symptoms)
        probabilities = knn_model.predict_proba(normalized_symptoms)[0]
        
        # Display results
        st.success(f"Predicted Diagnosis: **{prediction[0]}**")
        
        # Display probabilities
        st.subheader("Diagnosis Probabilities")
        
        # Create a DataFrame for the probabilities
        prob_df = pd.DataFrame({
            'Diagnosis': knn_model.classes_,
            'Probability': probabilities
        }).sort_values('Probability', ascending=False)
        
        # Plot probabilities
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = sns.barplot(x='Probability', y='Diagnosis', data=prob_df, ax=ax)
        
        # Add percentage labels
        for i, p in enumerate(probabilities):
            bars.text(p + 0.01, i, f'{p:.2%}', va='center')
            
        plt.xlim(0, 1.1)
        plt.title('Diagnosis Probabilities')
        st.pyplot(fig)

elif page == "Model Information":
    st.header("Model Information")
    
    st.subheader("Model Type")
    st.write("K-Nearest Neighbors (KNN) Classifier")
    
    st.subheader("Model Parameters")
    st.write(f"- Number of neighbors (k): 6")
    st.write(f"- Distance metric: Euclidean")
    
    # Display model performance if available
    if 'dataset' in model_data:
        from sklearn.model_selection import cross_val_score
        
        X = dataset.iloc[:, 0:28]
        y = dataset.iloc[:, 28]
        X_normalized = scaler.transform(X)
        
        cv_scores = cross_val_score(knn_model, X_normalized, y, cv=5)
        
        st.subheader("Model Performance")
        st.write(f"Cross-validation accuracy: {cv_scores.mean():.4f}")
        
        # Display individual fold scores
        st.write("Individual fold scores:")
        for i, score in enumerate(cv_scores):
            st.write(f"- Fold {i+1}: {score:.4f}")

elif page == "Data Visualization":
    st.header("Data Visualization")
    
    if 'dataset' in model_data:
        # Diagnosis distribution
        st.subheader("Diagnosis Distribution")
        diagnosis_counts = dataset['Diagnosis'].value_counts()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = sns.barplot(x=diagnosis_counts.index, y=diagnosis_counts.values, ax=ax)
        plt.title('Distribution of Dental Diagnoses')
        plt.xlabel('Diagnosis')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        
        for i, v in enumerate(diagnosis_counts.values):
            ax.text(i, v + 0.5, str(v), ha='center')
            
        plt.tight_layout()
        st.pyplot(fig)
        
        # Correlation heatmap
        st.subheader("Symptom Correlation Heatmap")
        X = dataset.iloc[:, 0:28]
        
        fig, ax = plt.subplots(figsize=(12, 10))
        corr = X.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=False, cmap='coolwarm', 
                    linewidths=0.5, vmin=-1, vmax=1, ax=ax)
        plt.title('Correlation Between Symptoms')
        plt.tight_layout()
        st.pyplot(fig)
        
        # Feature importance
        st.subheader("Top 10 Most Important Symptoms")
        
        from sklearn.ensemble import RandomForestClassifier
        
        X = dataset.iloc[:, 0:28]
        y = dataset.iloc[:, 28]
        X_normalized = scaler.transform(X)
        
        rf = RandomForestClassifier(random_state=42)
        rf.fit(X_normalized, y)
        
        feature_importance = pd.DataFrame({
            'Feature': symptom_names,
            'Importance': rf.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=feature_importance.head(10), ax=ax)
        plt.title('Top 10 Most Important Symptoms')
        plt.tight_layout()
        st.pyplot(fig)

# Footer
st.sidebar.markdown("---")
st.sidebar.info("Dental Diagnosis Expert System © 2023")
