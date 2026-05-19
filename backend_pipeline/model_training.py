from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os

def train_and_evaluate_models(X, y):
    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("\n--- Training Random Forest (Medical Triage) ---")
    # Using max_depth to prevent RAM overload on 1M+ rows
    rf_model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, class_weight='balanced', n_jobs=-1)
    rf_model.fit(X_train, y_train)
    rf_predictions = rf_model.predict(X_test)
    
    print("Mortality Prediction Confusion Matrix:")
    print(confusion_matrix(y_test, rf_predictions))
    print("\nClassification Report:")
    print(classification_report(y_test, rf_predictions))
    
    print("Saving AI Triage model to disk...")
    models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))
    os.makedirs(models_dir, exist_ok=True)
    
    model_path = os.path.join(models_dir, 'covid_risk_model.pkl')
    joblib.dump(rf_model, model_path)
    print(f"Model saved successfully at {model_path}")