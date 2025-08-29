"""
Configuration settings for AquaMind Water Quality Analyzer
==========================================================
This file contains configuration constants and settings for the application.
"""

import os

# File paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "WQD.csv")

# GUI Configuration
WINDOW_TITLE = "AquaMind Water Safety Checker v1.0"
WINDOW_SIZE = "900x700"
APP_VERSION = "1.0"

# Model Configuration
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_ESTIMATORS = 100
MAX_DEPTH = 10

# Color scheme
COLORS = {
    'primary': "#4CAF50",      # Green for analyze button
    'danger': "#f44336",       # Red for clear button  
    'warning': "#ff9800",      # Orange for clear output
    'info': "blue",            # Blue for headers
    'success': "darkgreen",    # Dark green for output header
    'text': "gray",            # Gray for secondary text
    'background': "#f8f8f8"    # Light gray for output background
}

# Water Quality Classifications
QUALITY_CLASSIFICATIONS = {
    0: {
        'label': 'SAFE',
        'emoji': '✅',
        'message': 'Water is SAFE to use.',
        'description': 'All parameters are within safe ranges for farming and suitable for aquatic life.'
    },
    1: {
        'label': 'CAUTIONARY', 
        'emoji': '⚠️',
        'message': 'Water quality is CAUTIONARY.',
        'description': 'Some parameters are at borderline levels. Consider treatment before use.'
    },
    2: {
        'label': 'UNSAFE',
        'emoji': '🔴', 
        'message': 'Water is UNSAFE!',
        'description': 'Critical contamination levels detected. Do NOT use this water for farming or for aquatic life.'
    }
}

# Application metadata
APP_INFO = {
    'name': 'AquaMind Water Quality Analyzer',
    'version': APP_VERSION,
    'author': 'Karthik Ravuru',
    'description': 'AI-powered water quality analysis tool',
    'emoji': '🧪'
}
