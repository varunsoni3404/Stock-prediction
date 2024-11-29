import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

class StockPredictor:
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.scaler = StandardScaler()
        
    def prepare_features(self, df):
        """Prepare features for the model."""
        feature_columns = [
            'Open', 'High', 'Low', 'Close', 'Volume',  # Stock price features
            'returns', 'volatility',                    # Technical indicators
            'ma5', 'ma20',                             # Moving averages
            'post_count', 'avg_score', 'avg_comments'   # Reddit features
        ]
        
        # Ensure all feature columns exist
        missing_cols = [col for col in feature_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns in data: {missing_cols}")
            
        # Select features and handle any remaining NaN values
        X = df[feature_columns].fillna(0)
        y = df['target']
        
        return X, y
    
    def train(self, training_data):
        """Train the model."""
        X, y = self.prepare_features(training_data)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train the model
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate the model
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)
        
        y_pred = self.model.predict(X_test_scaled)
        
        # Get feature importance
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        evaluation = {
            'train_score': train_score,
            'test_score': test_score,
            'classification_report': classification_report(y_test, y_pred, zero_division=0),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'feature_importance': feature_importance
        }
        
        return evaluation
    
    def predict(self, features):
        """Make predictions for new data."""
        features_scaled = self.scaler.transform(features)
        predictions = self.model.predict(features_scaled)
        probabilities = self.model.predict_proba(features_scaled)
        
        return predictions, probabilities
    
    def save_model(self, path='models/stock_predictor.joblib'):
        """Save the trained model."""
        model_data = {
            'model': self.model,
            'scaler': self.scaler
        }
        joblib.dump(model_data, path)
    
    def load_model(self, path='models/stock_predictor.joblib'):
        """Load a trained model."""
        model_data = joblib.load(path)
        self.model = model_data['model']
        self.scaler = model_data['scaler'] 