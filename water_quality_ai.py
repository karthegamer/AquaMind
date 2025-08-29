"""
Water Quality AI Core Module
============================
This module contains all the machine learning logic for water quality analysis.
It handles data loading, model training, prediction, and evaluation.

Author: [Your Name]
Date: August 2025
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os
from typing import Tuple, List, Optional, Callable
from dataclasses import dataclass, field
from config import QUALITY_CLASSIFICATIONS


@dataclass
class WaterQualityParameters:
    """
    Data class representing water quality measurements with dataset-average default values.
    
    Default values are calculated from the actual dataset averages, making them
    statistically representative of real water samples.
    """
    # These will be set dynamically from dataset averages
    temperature: float = 20.0        # Will be updated from dataset
    turbidity: float = 1.0           # Will be updated from dataset
    dissolved_oxygen: float = 8.0    # Will be updated from dataset
    bod: float = 2.0                 # Will be updated from dataset
    co2: float = 10.0                # Will be updated from dataset
    ph: float = 7.0                  # Will be updated from dataset
    alkalinity: float = 120.0        # Will be updated from dataset
    hardness: float = 150.0          # Will be updated from dataset
    calcium: float = 60.0            # Will be updated from dataset
    ammonia: float = 0.1             # Will be updated from dataset
    nitrite: float = 0.01            # Will be updated from dataset
    phosphorus: float = 0.05         # Will be updated from dataset
    h2s: float = 0.001               # Will be updated from dataset
    plankton: float = 50.0           # Will be updated from dataset
    
    @classmethod
    def create_from_dataset_averages(cls, dataset_path: str) -> 'WaterQualityParameters':
        """
        Create a WaterQualityParameters instance with averages calculated from the dataset.
        
        Args:
            dataset_path: Path to the water quality CSV dataset
            
        Returns:
            WaterQualityParameters instance with dataset-average defaults
        """
        import pandas as pd
        
        try:
            # Load the dataset
            data = pd.read_csv(dataset_path)
            
            # Calculate means for each feature (excluding the target column)
            features = data.drop('Water Quality', axis=1, errors='ignore')
            
            # Get column names and map them to our parameter names
            column_mapping = {
                'Temp': 'temperature',
                'Temperature': 'temperature', 
                'Turbidity': 'turbidity',
                'DO': 'dissolved_oxygen',
                'DO(mg/L)': 'dissolved_oxygen',
                'BOD': 'bod',
                'BOD (mg/L)': 'bod',
                'CO2': 'co2',
                'pH': 'ph',
                'Alkalinity': 'alkalinity',
                'Alkalinity (mg L-1 )': 'alkalinity',
                'Hardness': 'hardness', 
                'Hardness (mg L-1 )': 'hardness',
                'Calcium': 'calcium',
                'Calcium (mg L-1 )': 'calcium',
                'Ammonia': 'ammonia',
                'Ammonia (mg L-1 )': 'ammonia',
                'Nitrite': 'nitrite',
                'Nitrite (mg L-1 )': 'nitrite',
                'Phosphorus': 'phosphorus',
                'Phosphorus (mg L-1 )': 'phosphorus',
                'H2S': 'h2s',
                'H2S (mg L-1 )': 'h2s',
                'Plankton': 'plankton',
                'Plankton (No. L-1)': 'plankton'
            }
            
            # Calculate averages
            # Filter rows where Water Quality == 1
            filtered_data = data[data['Water Quality'] == 1]
            filtered_features = filtered_data.drop('Water Quality', axis=1, errors='ignore')

            averages = {}
            for col in filtered_features.columns:
                # Find matching parameter name
                param_name = None
                for dataset_col, param_col in column_mapping.items():
                    if dataset_col.lower() in col.lower():
                        param_name = param_col
                        break

                if param_name:
                    # Calculate mean, handling NaN values
                    avg_value = filtered_features[col].mean()
                    if not pd.isna(avg_value):
                        averages[param_name] = round(avg_value, 2)
            
            # Create instance with calculated averages
            return cls(
                temperature=averages.get('temperature', 20.0),
                turbidity=averages.get('turbidity', 1.0),
                dissolved_oxygen=averages.get('dissolved_oxygen', 8.0),
                bod=averages.get('bod', 2.0),
                co2=averages.get('co2', 10.0),
                ph=averages.get('ph', 7.0),
                alkalinity=averages.get('alkalinity', 120.0),
                hardness=averages.get('hardness', 150.0),
                calcium=averages.get('calcium', 60.0),
                ammonia=averages.get('ammonia', 0.1),
                nitrite=averages.get('nitrite', 0.01),
                phosphorus=averages.get('phosphorus', 0.05),
                h2s=averages.get('h2s', 0.001),
                plankton=averages.get('plankton', 50.0)
            )
            
        except Exception as e:
            print(f"Warning: Could not calculate dataset averages: {e}")
            print("Using fallback default values instead.")
            return cls()  # Return with original defaults
    
    def to_list(self) -> List[float]:
        """Convert parameters to list format for ML model prediction."""
        return [
            self.temperature, self.turbidity, self.dissolved_oxygen, self.bod,
            self.co2, self.ph, self.alkalinity, self.hardness, self.calcium,
            self.ammonia, self.nitrite, self.phosphorus, self.h2s, self.plankton
        ]
    
    def from_dict(self, data: dict) -> 'WaterQualityParameters':
        """Create instance from dictionary of values."""
        return WaterQualityParameters(
            temperature=data.get('temperature', self.temperature),
            turbidity=data.get('turbidity', self.turbidity),
            dissolved_oxygen=data.get('dissolved_oxygen', self.dissolved_oxygen),
            bod=data.get('bod', self.bod),
            co2=data.get('co2', self.co2),
            ph=data.get('ph', self.ph),
            alkalinity=data.get('alkalinity', self.alkalinity),
            hardness=data.get('hardness', self.hardness),
            calcium=data.get('calcium', self.calcium),
            ammonia=data.get('ammonia', self.ammonia),
            nitrite=data.get('nitrite', self.nitrite),
            phosphorus=data.get('phosphorus', self.phosphorus),
            h2s=data.get('h2s', self.h2s),
            plankton=data.get('plankton', self.plankton)
        )
    
    def validate(self) -> List[str]:
        """Validate parameters and return list of issues if any."""
        issues = []
        
        if self.temperature < -10 or self.temperature > 50:
            issues.append("Temperature should be between -10°C and 50°C")
        if self.turbidity < 0:
            issues.append("Turbidity cannot be negative")
        if self.ph < 0 or self.ph > 14:
            issues.append("pH should be between 0 and 14")
        if self.dissolved_oxygen < 0:
            issues.append("Dissolved oxygen cannot be negative")
        
        return issues


@dataclass
class ModelConfiguration:
    """Configuration settings for the machine learning model."""
    random_state: int = 42
    test_size: float = 0.2
    n_estimators: int = 100
    max_depth: int = 10
    class_weight: str = 'balanced'
    stratify: bool = True


@dataclass
class PredictionResult:
    """Result of water quality prediction with all relevant information."""
    prediction_class: int
    confidence_scores: List[float] = field(default_factory=list)
    result_message: str = ""
    detailed_notes: str = ""
    parameters_used: Optional[WaterQualityParameters] = None
    timestamp: str = field(default_factory=lambda: pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    @property
    def is_safe(self) -> bool:
        """Check if water is classified as safe."""
        return self.prediction_class == 0
    
    @property
    def risk_level(self) -> str:
        """Get human-readable risk level."""
        classification = QUALITY_CLASSIFICATIONS.get(self.prediction_class, {'label': 'Unknown'})
        return classification['label']


class WaterQualityAI:
    """
    A machine learning model for analyzing water quality and predicting safety levels.
    
    The model classifies water into three categories:
    - 0: SAFE - Water meets all safety standards
    - 1: CAUTIONARY - Some parameters are borderline, use with care  
    - 2: UNSAFE - Critical values detected, do not use
    """
    
    def __init__(self, data_path: str, output_callback: Optional[Callable] = None, 
                 config: Optional[ModelConfiguration] = None):
        """
        Initialize the Water Quality AI model.
        
        Args:
            data_path: Path to the water quality dataset CSV file
            output_callback: Optional function to call for output messages (for GUI integration)
            config: Model configuration (uses defaults if not provided)
        """
        self.data_path = data_path
        self.config = config or ModelConfiguration()
        self.model = None
        self.feature_names = None
        self.test_accuracy = 0.0
        self.feature_importance = None
        self.output_callback = output_callback or print
        self.dataset_averages = None  # Will store calculated averages
        
        # Water quality parameter information
        self.parameter_info = {
            'Temperature': {'unit': '°C', 'description': 'Water temperature'},
            'Turbidity': {'unit': 'NTU', 'description': 'Water clarity measurement'},
            'DO': {'unit': 'mg/L', 'description': 'Dissolved Oxygen content'},
            'BOD': {'unit': 'mg/L', 'description': 'Biochemical Oxygen Demand'},
            'CO2': {'unit': 'mg/L', 'description': 'Carbon Dioxide content'},
            'pH': {'unit': '0-14', 'description': 'Acidity/alkalinity level'},
            'Alkalinity': {'unit': 'mg/L', 'description': "Water's buffering capacity"},
            'Hardness': {'unit': 'mg/L', 'description': 'Mineral content (Ca2+ and Mg2+)'},
            'Calcium': {'unit': 'mg/L', 'description': 'Calcium ion concentration'},
            'Ammonia': {'unit': 'mg/L', 'description': 'Ammonia/Ammonium content'},
            'Nitrite': {'unit': 'mg/L', 'description': 'Nitrite nitrogen content'},
            'Phosphorus': {'unit': 'mg/L', 'description': 'Phosphate content'},
            'H2S': {'unit': 'mg/L', 'description': 'Hydrogen Sulfide content'},
            'Plankton': {'unit': 'cells/mL', 'description': 'Microorganism count'}
        }
    
    def _log(self, message: str) -> None:
        """Log a message using the output callback."""
        self.output_callback(message)
    
    def load_and_train_model(self) -> Tuple[float, dict]:
        """
        Loads water quality dataset, preprocesses it, and trains a Random Forest classifier.
        
        Returns:
            Tuple of (test_accuracy, feature_importance_dict)
        """
        self._log("Loading water quality dataset...")
        
        # Verify file exists
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Dataset file not found: {self.data_path}")
        
        # Load the water quality dataset
        data = pd.read_csv(self.data_path)
        self._log("Dataset loaded successfully!")
        self._log(f"Dataset shape: {data.shape}")
        self._log(f"Columns in dataset: {data.columns.tolist()}")
        
        # Shuffle the dataset for better training
        data = data.sample(frac=1, random_state=42)
        self._log("Dataset shuffled for random distribution.")
        
        # Calculate and store dataset averages for default values
        self._calculate_dataset_averages(data)
        
        # Separate features from target variable
        features = data.drop('Water Quality', axis=1)
        labels = data['Water Quality']
        
        self.feature_names = features.columns.tolist()
        
        self._log(f"Features shape: {features.shape}")
        self._log(f"Labels distribution:\n{labels.value_counts().sort_index()}")
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, 
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=labels if self.config.stratify else None
        )
        
        self._log(f"Training set size: {X_train.shape[0]} samples")
        self._log(f"Testing set size: {X_test.shape[0]} samples")
        
        # Initialize and train the model
        self.model = RandomForestClassifier(
            random_state=self.config.random_state,
            class_weight=self.config.class_weight,
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth
        )
        
        self._log("Training Random Forest model...")
        self.model.fit(X_train, y_train)
        self._log("Model training completed!")
        
        # Evaluate the model
        self.test_accuracy = self.model.score(X_test, y_test)
        self._log(f"Model Test Accuracy: {self.test_accuracy:.4f} ({self.test_accuracy*100:.2f}%)")
        
        # Get feature importance
        self.feature_importance = pd.DataFrame({
            'feature': features.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        self._log("\nTop 5 Most Important Features:")
        for idx, row in self.feature_importance.head().iterrows():
            self._log(f"  {row['feature']}: {row['importance']:.4f}")
        
        return self.test_accuracy, self.feature_importance.to_dict('records')
    
    def predict_water_quality(self, parameters) -> PredictionResult:
        """
        Predict water quality based on input parameters.
        
        Args:
            parameters: Either a list of 14 float values or WaterQualityParameters instance
            
        Returns:
            PredictionResult with comprehensive analysis information
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call load_and_train_model() first.")
        
        # Handle different input types
        if isinstance(parameters, WaterQualityParameters):
            param_obj = parameters
            param_list = parameters.to_list()
        elif isinstance(parameters, list):
            if len(parameters) != 14:
                raise ValueError(f"Expected 14 parameters, got {len(parameters)}")
            param_list = parameters
            param_obj = WaterQualityParameters(*parameters)
        else:
            raise ValueError("Parameters must be a list or WaterQualityParameters instance")
        
        # Validate parameters
        validation_issues = param_obj.validate()
        if validation_issues:
            self._log(f"Parameter validation warnings: {'; '.join(validation_issues)}")
        
        # Make prediction with confidence scores
        prediction = self.model.predict([param_list])[0]
        probabilities = self.model.predict_proba([param_list])[0].tolist()
        
        # Get classification info from config
        classification = QUALITY_CLASSIFICATIONS.get(prediction, {
            'emoji': '❓',
            'message': 'Unknown classification',
            'description': 'Unable to classify water quality'
        })
        
        # Format result message with emoji
        result = f"{classification['emoji']} {classification['message']}"
        notes = classification['description']
        
        return PredictionResult(
            prediction_class=prediction,
            confidence_scores=probabilities,
            result_message=result,
            detailed_notes=notes,
            parameters_used=param_obj
        )
    
    def _calculate_dataset_averages(self, data: pd.DataFrame) -> None:
        """
        Calculate average values from the dataset and store them.
        
        Args:
            data: The loaded dataset DataFrame
        """
        try:
            # Get features (exclude target column)
            features = data.drop('Water Quality', axis=1, errors='ignore')
            
            # Calculate means for each column
            self.dataset_averages = {}
            for col in features.columns:
                avg_value = features[col].mean()
                if not pd.isna(avg_value):
                    self.dataset_averages[col] = round(avg_value, 3)
            
            self._log("Dataset averages calculated:")
            for col, avg in self.dataset_averages.items():
                self._log(f"  {col}: {avg}")
                
        except Exception as e:
            self._log(f"Warning: Could not calculate dataset averages: {e}")
            self.dataset_averages = {}
    
    def create_default_parameters(self) -> WaterQualityParameters:
        """Create a WaterQualityParameters instance with dataset-average default values."""
        # Use the dataset path to create parameters with actual dataset averages
        return WaterQualityParameters.create_from_dataset_averages(self.data_path)
    
    def get_parameter_info(self) -> dict:
        """Get information about water quality parameters."""
        return self.parameter_info
    
    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """Get feature importance rankings if model is trained."""
        return self.feature_importance
    
    def get_model_accuracy(self) -> float:
        """Get the model's test accuracy."""
        return self.test_accuracy
    
    def is_trained(self) -> bool:
        """Check if the model has been trained."""
        return self.model is not None


# Convenience function for backwards compatibility
def create_water_quality_ai(data_path: str, output_callback: Optional[Callable] = None) -> WaterQualityAI:
    """
    Create and return a WaterQualityAI instance.
    
    Args:
        data_path: Path to the dataset
        output_callback: Optional callback for output messages
        
    Returns:
        WaterQualityAI instance
    """
    return WaterQualityAI(data_path, output_callback)
