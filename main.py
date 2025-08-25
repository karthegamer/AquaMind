"""
AquaMind Water Quality Analyzer - Main Entry Point
==================================================
This is the main entry point for the AquaMind Water Quality Analyzer application.
It orchestrates the initialization of the AI model and GUI components.

Usage:
    python main.py

Author: Karthik Ravuru
Date: August 2025
"""

import sys
import os
from gui_application import create_application
from config import DATA_FILE, APP_INFO


def main():
    """Main application entry point."""
    print("="*60)
    print(f"🌊 Starting {APP_INFO['name']} v{APP_INFO['version']} 🌊")
    print("="*60)
    
    try:
        # Verify data file exists
        if not os.path.exists(DATA_FILE):
            print(f"ERROR: Data file not found: {DATA_FILE}")
            print("Please ensure WQD.csv is in the same directory as this script.")
            sys.exit(1)
        
        # Create and configure the application
        print("Initializing application...")
        app = create_application(DATA_FILE)
        
        # Create the GUI
        print("Creating GUI...")
        root = app.create_gui()
        
        # Train the model (this will happen in the GUI with output display)
        print("Training AI model (progress will be shown in GUI)...")
        app.train_model_and_update_gui()
        
        print("Application ready! GUI window should now be visible.")
        print("You can close this terminal - the application will continue running.")
        print("="*60)
        
        # Start the GUI event loop
        app.run()
        
    except Exception as e:
        print(f"ERROR: Failed to start application: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("AquaMind application closed. Thank you for using our water quality analyzer!")


if __name__ == "__main__":
    main()
