import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pickle
import os
import sys

def load_or_train_model():
    """Load existing model or train a new one"""
    # Define column names
    names = ['GP01','GP02','GP03','GP04','GP05', 'GP06','GP07', 'GP08', 'GP09', 'GP10', 
             'GP11', 'GP12', 'GP13', 'GP14', 'GP15', 'GP16', 'GP17', 'GP18', 'GP19', 'GP20', 
             'GP21', 'GP22', 'GP23', 'GP24', 'GP25','GP26','GP27','GP28', 'Diagnosis']
    
    # Check if model exists
    if os.path.exists('dental_model.pkl'):
        print("Loading existing model...")
        with open('dental_model.pkl', 'rb') as file:
            return pickle.load(file)
    
    print("Training new model...")
    # If not, train model
    try:
        dataset = pd.read_excel('rm1.xls', names=names)
    except FileNotFoundError:
        print("Error: Dataset file 'rm1.xls' not found.")
        sys.exit(1)
    
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
    
    print("Model trained and saved successfully.")
    return model_data

def predict_diagnosis(model_data):
    """Interactive function to predict diagnosis based on user input"""
    print("\n===== Dental Diagnosis Prediction Tool =====")
    print("Enter symptom values (0.0-1.0) where:")
    print("0.0 = Not happening, 0.25 = Undecided, 0.5 = Maybe")
    print("0.75 = Most likely, 1.0 = Sure\n")
    
    knn_model = model_data['model']
    scaler = model_data['scaler']
    symptom_names = model_data['symptom_names']
    
    # Get user input for each symptom
    user_symptoms = []
    for symptom in symptom_names:
        while True:
            try:
                value = float(input(f"{symptom} (0.0-1.0): "))
                if 0 <= value <= 1:
                    user_symptoms.append(value)
                    break
                else:
                    print("Value must be between 0 and 1")
            except ValueError:
                print("Please enter a valid number")
    
    # Normalize input
    normalized_symptoms = scaler.transform([user_symptoms])
    
    # Make prediction
    prediction = knn_model.predict(normalized_symptoms)
    probabilities = knn_model.predict_proba(normalized_symptoms)[0]
    
    # Display results
    print("\n===== Diagnosis Results =====")
    print(f"Predicted diagnosis: {prediction[0]}")
    print("\nProbability for each diagnosis:")
    
    # Sort probabilities in descending order
    sorted_indices = probabilities.argsort()[::-1]
    classes = knn_model.classes_
    
    for i in sorted_indices:
        print(f"{classes[i]}: {probabilities[i]:.4f} ({probabilities[i]*100:.2f}%)")
    
    return prediction[0]

def show_model_info(model_data):
    """Display information about the model"""
    print("\n===== Model Information =====")
    print("Model Type: K-Nearest Neighbors (KNN) Classifier")
    print("Model Parameters:")
    print("- Number of neighbors (k): 6")
    print("- Distance metric: Euclidean")
    
    if 'dataset' in model_data:
        from sklearn.model_selection import cross_val_score
        
        dataset = model_data['dataset']
        knn_model = model_data['model']
        scaler = model_data['scaler']
        
        X = dataset.iloc[:, 0:28]
        y = dataset.iloc[:, 28]
        X_normalized = scaler.transform(X)
        
        print("\nCalculating cross-validation scores...")
        cv_scores = cross_val_score(knn_model, X_normalized, y, cv=5)
        
        print(f"Cross-validation accuracy: {cv_scores.mean():.4f}")
        print("Individual fold scores:")
        for i, score in enumerate(cv_scores):
            print(f"- Fold {i+1}: {score:.4f}")

def main_menu():
    """Display the main menu and handle user choices"""
    # Load or train the model
    model_data = load_or_train_model()
    
    while True:
        print("\n===== Dental Diagnosis Expert System =====")
        print("1. Diagnose a new case")
        print("2. View model information")
        print("3. Exit")
        
        choice = input("\nEnter your choice (1-3): ")
        
        if choice == '1':
            predict_diagnosis(model_data)
        elif choice == '2':
            show_model_info(model_data)
        elif choice == '3':
            print("Thank you for using the Dental Diagnosis Expert System. Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main_menu()