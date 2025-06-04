import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.ion()  # Enable interactive mode for matplotlib
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
import seaborn as sns

# Add this code at the beginning of your script to check the current working directory
import os
print("Current working directory:", os.getcwd())
print("Files in current directory:", os.listdir())

# Define column names
names = ['GP01','GP02','GP03','GP04','GP05', 'GP06','GP07', 'GP08', 'GP09', 'GP10', 
         'GP11', 'GP12', 'GP13', 'GP14', 'GP15', 'GP16', 'GP17', 'GP18', 'GP19', 'GP20', 
         'GP21', 'GP22', 'GP23', 'GP24', 'GP25','GP26','GP27','GP28', 'Diagnosis']

# Use the correct filename: rm1.xls (not rml.xls)
Dataset = pd.read_excel('rm1.xls', names=names)

# Data visualization
print("\nDiagnosis distribution:")
print(Dataset['Diagnosis'].value_counts())

# Data preprocessing
X = Dataset.iloc[:, 0:28]
y = Dataset.iloc[:, 28]

# Visualize data distribution
plt.figure(figsize=(12, 6))
diagnosis_counts = Dataset['Diagnosis'].value_counts()
ax = sns.barplot(x=diagnosis_counts.index, y=diagnosis_counts.values)
plt.title('Distribution of Dental Diagnoses')
plt.xlabel('Diagnosis')
plt.ylabel('Count')
plt.xticks(rotation=45)
for i, v in enumerate(diagnosis_counts.values):
    ax.text(i, v + 0.5, str(v), ha='center')
plt.tight_layout()
plt.show()

# Correlation heatmap of symptoms
plt.figure(figsize=(14, 12))
corr = X.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=False, cmap='coolwarm', 
            linewidths=0.5, vmin=-1, vmax=1)
plt.title('Correlation Between Symptoms')
plt.tight_layout()
plt.show()

# Add data normalization
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)

# Use normalized data for training
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=1)

# Train model
knnmodel = KNeighborsClassifier(n_neighbors=6, metric='euclidean')
knnmodel.fit(X_train, y_train)

# Predict
y_predict1 = knnmodel.predict(X_test)

# Output visualization
prediction_output = pd.DataFrame(data=[y_predict1, y_test.values], 
                                index=['Predicted Output', 'Actual Output'])
print("\nPrediction counts:")
print(prediction_output.iloc[0,:].value_counts())

# Calculate accuracy
from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, y_predict1)
print(f"\nAccuracy: {acc:.4f}")

# Confusion matrix
from sklearn.metrics import confusion_matrix, classification_report
cm = confusion_matrix(y_test.values, y_predict1)
cr = classification_report(y_test.values, y_predict1)
cm1 = pd.DataFrame(data=cm, 
                  index=['Pulpitis', 'Stomatitis', 'Periodontitis', 'Karies Gigi', 'Abses gigi', 'Gingivitis'],
                  columns=['Pulpitis', 'Stomatitis', 'Periodontitis', 'Karies Gigi', 'Abses gigi', 'Gingivitis'])

print("\nConfusion Matrix:")
print(cm1)
print("\nClassification Report:")
print(cr)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm1, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
plt.pause(10)  # Pause for 10 seconds to allow user to view the plot

# K-fold cross-validation
from sklearn.model_selection import cross_val_score
cv_scores = cross_val_score(knnmodel, X, y, cv=5)
print(f"\nCross-validation scores: {cv_scores}")
print(f"Average CV score: {cv_scores.mean():.4f}")

# Hyperparameter tuning
from sklearn.model_selection import GridSearchCV
param_grid = {'n_neighbors': range(1, 15), 'metric': ['euclidean', 'manhattan']}
grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
grid_search.fit(X, y)
print(f"\nBest parameters: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_:.4f}")

# Feature importance using Random Forest
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
feature_importance = pd.DataFrame({
    'Feature': names[:-1],  # Exclude 'Diagnosis'
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nTop 10 most important symptoms:")
print(feature_importance.head(10))

# Visualize feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance.head(10))
plt.title('Top 10 Most Important Symptoms')
plt.tight_layout()
plt.show()

def predict_diagnosis():
    """Interactive function to predict diagnosis based on user input"""
    print("\n=== Dental Diagnosis Prediction Tool ===")
    print("Enter symptom values (0.0-1.0) where:")
    print("0.0 = Not happening, 0.25 = Undecided, 0.5 = Maybe")
    print("0.75 = Most likely, 1.0 = Sure\n")
    
    # Get user input for each symptom
    user_symptoms = []
    for i, symptom in enumerate(names[:-1]):
        while True:
            try:
                value = float(input(f"{symptom}: "))
                if 0 <= value <= 1:
                    user_symptoms.append(value)
                    break
                else:
                    print("Value must be between 0 and 1")
            except ValueError:
                print("Please enter a valid number")
    
    # Normalize input if your model was trained on normalized data
    if 'scaler' in globals():
        user_symptoms = scaler.transform([user_symptoms])
    else:
        user_symptoms = [user_symptoms]
    
    # Make prediction
    prediction = knnmodel.predict(user_symptoms)
    probability = knnmodel.predict_proba(user_symptoms)
    
    print(f"\nPredicted diagnosis: {prediction[0]}")
    print("\nProbability for each diagnosis:")
    for i, diagnosis in enumerate(knnmodel.classes_):
        print(f"{diagnosis}: {probability[0][i]:.4f}")

# Call the function to start the interactive tool
# predict_diagnosis()  # Uncomment to use

# Compare multiple classification models
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

# Define models to compare
models = {
    'KNN': KNeighborsClassifier(n_neighbors=6),
    'Random Forest': RandomForestClassifier(),
    'SVM': SVC(probability=True),
    'Naive Bayes': GaussianNB(),
    'Logistic Regression': LogisticRegression(max_iter=1000)
}

# Train and evaluate each model
results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    results.append({'Model': name, 'Accuracy': accuracy})

# Display results
results_df = pd.DataFrame(results)
print("\nModel Comparison:")
print(results_df)

# Plot model comparison
plt.figure(figsize=(10, 6))
sns.barplot(x='Model', y='Accuracy', data=results_df)
plt.title('Model Accuracy Comparison')
plt.ylim(0, 1)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


