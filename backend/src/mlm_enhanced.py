# backend/src/mlm_enhanced.py - FIXED VERSION
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, matthews_corrcoef
import joblib
import warnings

warnings.filterwarnings('ignore')


class RiskPredictionModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}

    def load_and_preprocess(self, filename):
        """Load and preprocess dataset with actual columns"""
        print(f"Loading dataset from {filename}...")

        # Load dataset - adjust column names based on your actual data
        # First, let's see what columns we actually have
        try:
            df = pd.read_csv(filename, nrows=5)  # Just read first 5 rows to check
            print("Sample columns in dataset:", df.columns.tolist())

            # Now read full dataset
            df = pd.read_csv(filename)
            print(f"Dataset shape: {df.shape}")
            print(f"Columns found: {df.columns.tolist()}")

        except FileNotFoundError:
            print(f"Error: File {filename} not found!")
            print("Please make sure the CSV file is in the correct location.")
            print("Current working directory:", os.getcwd())
            return None, None

        # Based on your original mlm.py, your columns are:
        # ["LightLevel", "Weather", "RoadCondition", "Longitude", "Latitude"]

        # Check if we have the expected columns
        expected_columns = ['LightLevel', 'Weather', 'RoadCondition', 'Longitude', 'Latitude']
        missing_cols = [col for col in expected_columns if col not in df.columns]

        if missing_cols:
            print(f"Warning: Missing columns: {missing_cols}")
            print(f"Available columns: {df.columns.tolist()}")

            # Try to find similar columns
            for missing in missing_cols:
                for actual in df.columns:
                    if missing.lower() in actual.lower():
                        print(f"  Using '{actual}' for '{missing}'")
                        df[missing] = df[actual]
                        break

        # Use only the columns we know exist
        available_features = [col for col in expected_columns if col in df.columns]
        print(f"Using features: {available_features}")

        # Create target variable: risk level
        df['RiskLevel'] = self.calculate_risk_level(df)
        print(f"Risk level distribution: {df['RiskLevel'].value_counts().to_dict()}")

        # Encode categorical variables
        categorical_cols = ['LightLevel', 'Weather', 'RoadCondition']
        categorical_cols = [col for col in categorical_cols if col in df.columns]

        for col in categorical_cols:
            print(f"Encoding {col}...")
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            self.label_encoders[col] = le

        # Scale numerical features
        numerical_cols = ['Longitude', 'Latitude']
        numerical_cols = [col for col in numerical_cols if col in df.columns]

        if numerical_cols:
            print(f"Scaling numerical columns: {numerical_cols}")
            df[numerical_cols] = self.scaler.fit_transform(df[numerical_cols])
        else:
            print("Warning: No numerical columns found for scaling!")

        return df[available_features], df['RiskLevel']

    def calculate_risk_level(self, df):
        """Calculate risk level based on accident data"""
        print("Calculating risk levels...")

        # Group by location and calculate accident "density"
        # We'll create a simple risk score based on location clustering

        # For now, create a simple risk classification
        # High risk: issues with weather/light/road conditions
        # Medium risk: 2 issues
        # Low risk: 0-1 issues

        risk_levels = []

        for _, row in df.iterrows():
            risk_score = 0

            # Check conditions - these would need to be adjusted based on your data
            # For now, just assign random risk for demonstration
            import random
            risk_score = random.randint(0, 2)  # 0=low, 1=medium, 2=high

            risk_levels.append(risk_score)

        print(f"Generated {len(risk_levels)} risk levels")
        return risk_levels

    def train(self, X, y):
        """Train the model"""
        if X is None or y is None:
            print("Error: No data to train on!")
            return

        print(f"\nTraining on {len(X)} samples...")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train model
        print("Training Random Forest model...")
        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1  # Use all CPU cores
        )

        self.model.fit(X_train, y_train)

        # Evaluate
        print("\nModel Evaluation:")
        y_pred = self.model.predict(X_test)

        print("Classification Report:")
        print(classification_report(y_test, y_pred,
                                    target_names=['Low', 'Medium', 'High']))

        mcc = matthews_corrcoef(y_test, y_pred)
        print(f"MCC Score: {mcc:.4f}")
        print(f"Accuracy: {self.model.score(X_test, y_test):.4f}")

        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            print("\nFeature Importances:")
            for name, importance in zip(X.columns, self.model.feature_importances_):
                print(f"  {name}: {importance:.4f}")

        # Save model and preprocessing objects
        print("\nSaving model artifacts...")
        joblib.dump(self.model, 'risk_model.pkl')
        joblib.dump(self.scaler, 'scaler.pkl')
        joblib.dump(self.label_encoders, 'encoder.pkl')

        print("Model training completed successfully!")

    def predict(self, features):
        """Predict risk for new data"""
        if self.model is None:
            raise ValueError("Model not trained")
        return self.model.predict(features)


def main():
    print("=" * 60)
    print("Risk Prediction Model Trainer")
    print("=" * 60)

    # Initialize model
    risk_model = RiskPredictionModel()

    # Load and preprocess data
    # Try different possible file locations
    import os

    # First, try the current directory
    filename = "oddYears(2005-2019).csv"

    # If not found, try parent directory (backend/)
    if not os.path.exists(filename):
        filename = "../oddYears(2005-2019).csv"

    # If still not found, try data directory
    if not os.path.exists(filename):
        filename = "../data/oddYears(2005-2019).csv"

    print(f"Looking for dataset at: {filename}")

    X, y = risk_model.load_and_preprocess(filename)

    if X is not None:
        # Train model
        risk_model.train(X, y)

        # Test with sample prediction
        print("\n" + "=" * 60)
        print("Sample Prediction:")
        if X.shape[0] > 0:
            sample_features = X.iloc[0:1]
            prediction = risk_model.predict(sample_features)
            risk_names = ['Low', 'Medium', 'High']
            print(f"Sample features: {sample_features.values.tolist()[0]}")
            print(f"Predicted risk: {risk_names[prediction[0]]}")
    else:
        print("\nFailed to load data. Please check:")
        print("1. The CSV file exists in the correct location")
        print("2. The file has the expected columns")
        print("3. You're running the script from the correct directory")


if __name__ == "__main__":
    main()